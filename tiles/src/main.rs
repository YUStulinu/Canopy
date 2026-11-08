//! canopy-tiles: map tiles of Sentinel-2 composites and of land cover predicted on the fly.
//!
//!   canopy-tiles [--data ../data] [--models ../models] [--port 8710]
//!
//!   GET /regions                               regions, grids, years
//!   GET /tiles/{rgb|classes}/{year}/{z}/{x}/{y}.png
//!   GET /classes/{region}/{year}               the class map: size x size bytes (255 = no data)
//!   GET /index/{region}/{year}/{nbr|ndvi}      a summer spectral index: size x size i16 (x 10000)
//!   GET /stats                                 inference timings

mod geo;
mod infer;
mod store;

use anyhow::Result;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use std::collections::HashSet;
use std::sync::Arc;
use store::{Store, CORE, NODATA};

const COLORS: [[u8; 3]; 10] = [
    [0x00, 0x64, 0x00], // tree cover
    [0xff, 0xbb, 0x22], // shrubland
    [0xff, 0xff, 0x4c], // grassland
    [0xf0, 0x96, 0xff], // cropland
    [0xfa, 0x00, 0x00], // built-up
    [0xb4, 0xb4, 0xb4], // bare / sparse vegetation
    [0xf0, 0xf0, 0xf0], // snow and ice
    [0x00, 0x64, 0xc8], // permanent water
    [0x00, 0x96, 0xa0], // herbaceous wetland
    [0xfa, 0xe6, 0xa0], // moss and lichen
];

type AppState = Arc<Store>;

fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = std::path::PathBuf::from(arg(&args, "--data", "../data"));
    let models = std::path::PathBuf::from(arg(&args, "--models", "../models"));
    let port: u16 = arg(&args, "--port", "8710").parse()?;
    let model = match infer::Model::load(&models) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("no model ({e:#}); serving imagery only");
            None
        }
    };
    let store = Arc::new(Store::open(&data, model)?);
    eprintln!("{} regions from {}", store.regions.len(), data.display());
    let app = Router::new()
        .route("/regions", get(regions))
        .route("/tiles/regions", get(regions))
        .route("/tiles/{layer}/{year}/{z}/{x}/{y}", get(tile))
        .route("/classes/{region}/{year}", get(classes))
        .route("/index/{region}/{year}/{name}", get(index))
        .route("/stats", get(stats))
        .with_state(store);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    eprintln!("listening on http://127.0.0.1:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn err(code: StatusCode, msg: impl ToString) -> Response {
    (code, [(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")], msg.to_string()).into_response()
}

async fn regions(State(s): State<AppState>) -> Response {
    let infos: Vec<_> = s.regions.iter().map(|r| r.info.clone()).collect();
    ([(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")], Json(infos)).into_response()
}

async fn stats(State(s): State<AppState>) -> Response {
    let mut t = s.inference_ms.lock().unwrap().clone();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| if t.is_empty() { 0.0 } else { t[((t.len() - 1) as f64 * q).round() as usize] };
    let body = serde_json::json!({"classes": s.model.as_ref().map(|m| m.classes.clone()), "blocks": t.len(), "block_pixels": CORE * CORE, "median_ms": pick(0.5), "p90_ms": pick(0.9)});
    ([(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")], Json(body)).into_response()
}

async fn classes(State(s): State<AppState>, Path((region, year)): Path<(String, i32)>) -> Response {
    let s2 = s.clone();
    let res = tokio::task::spawn_blocking(move || {
        let r = s2.region(&region).ok_or_else(|| anyhow::anyhow!("no region {region}"))?;
        let map = s2.class_map(r, year)?;
        Ok::<_, anyhow::Error>((r.info.size, map))
    })
    .await;
    match res {
        Ok(Ok((size, map))) => (
            [(header::CONTENT_TYPE, "application/octet-stream".to_string()), (header::HeaderName::from_static("x-size"), size.to_string())],
            map,
        )
            .into_response(),
        Ok(Err(e)) => err(StatusCode::NOT_FOUND, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// A summer spectral index over a whole region, as little-endian i16 (index x 10000; i16::MIN
/// where there is no data). `nbr`: (B08 - B12) / (B08 + B12), the normalised burn ratio, which
/// drops sharply where forest is cut; `ndvi`: (B08 - B04) / (B08 + B04).
pub fn spectral_index(comp: &store::Composite, name: &str) -> Option<Vec<i16>> {
    let (a, b) = match name {
        "nbr" => (16, 19),  // summer B08, B12
        "ndvi" => (16, 12), // summer B08, B04
        _ => return None,
    };
    let n = comp.size;
    let mut out = vec![i16::MIN; n * n];
    for r in 0..n {
        for c in 0..n {
            let (x, y) = (comp.at(a, r, c) as f32, comp.at(b, r, c) as f32);
            if x > 0.0 && y > 0.0 {
                out[r * n + c] = ((x - y) / (x + y) * 10000.0).round() as i16;
            }
        }
    }
    Some(out)
}

async fn index(State(s): State<AppState>, Path((region, year, name)): Path<(String, i32, String)>) -> Response {
    let res = tokio::task::spawn_blocking(move || {
        let r = s.region(&region).ok_or_else(|| anyhow::anyhow!("no region {region}"))?;
        let comp = s.composite(r, year)?;
        let v = spectral_index(&comp, &name).ok_or_else(|| anyhow::anyhow!("indices: nbr, ndvi"))?;
        Ok::<_, anyhow::Error>((r.info.size, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()))
    })
    .await;
    match res {
        Ok(Ok((size, bytes))) => (
            [(header::CONTENT_TYPE, "application/octet-stream".to_string()), (header::HeaderName::from_static("x-size"), size.to_string())],
            bytes,
        )
            .into_response(),
        Ok(Err(e)) => err(StatusCode::NOT_FOUND, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn tile(State(s): State<AppState>, Path((layer, year, z, x, y)): Path<(String, i32, u32, u32, String)>) -> Response {
    let Some(y) = y.strip_suffix(".png").and_then(|v| v.parse::<u32>().ok()) else {
        return err(StatusCode::BAD_REQUEST, "expected {y}.png");
    };
    if layer != "rgb" && layer != "classes" {
        return err(StatusCode::NOT_FOUND, "layers: rgb, classes");
    }
    let res = tokio::task::spawn_blocking(move || render(&s, &layer, year, z, x, y)).await;
    match res {
        Ok(Ok(png)) => ([(header::CONTENT_TYPE, "image/png"), (header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"), (header::CACHE_CONTROL, "max-age=3600")], png).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// A 256 x 256 RGBA tile: every pixel's centre is mapped into whichever region holds it.
fn render(s: &Store, layer: &str, year: i32, z: u32, x: u32, y: u32) -> Result<Vec<u8>> {
    let (w, so, e, n) = geo::tile_bounds(z, x, y);
    let hits: Vec<&store::Region> =
        s.regions.iter().filter(|r| r.info.years.contains(&year) && r.info.bbox[0] < e && r.info.bbox[2] > w && r.info.bbox[1] < n && r.info.bbox[3] > so).collect();
    let mut rgba = vec![0u8; 256 * 256 * 4];
    if !hits.is_empty() {
        // Where each pixel falls: (region index, column, row).
        let mut at = vec![None; 256 * 256];
        let mut needed = HashSet::new();
        for py in 0..256 {
            for px in 0..256 {
                let (lon, lat) = geo::tile_pixel(z, x, y, px as f64, py as f64);
                for (k, r) in hits.iter().enumerate() {
                    if let Some((c, rr)) = r.pixel(lon, lat) {
                        at[py * 256 + px] = Some((k, c, rr));
                        needed.insert((k, rr / CORE, c / CORE));
                        break;
                    }
                }
            }
        }
        if layer == "rgb" {
            let comps: Vec<_> = hits.iter().map(|r| s.composite(r, year)).collect::<Result<_>>()?;
            for (i, a) in at.iter().enumerate() {
                if let Some((k, c, r)) = *a {
                    let cp = &comps[k];
                    // Summer true colour: channels 12, 11, 10 (B04, B03, B02).
                    let v = [12, 11, 10].map(|ch| cp.at(ch, r, c));
                    if v[0] == 0 {
                        continue;
                    }
                    for j in 0..3 {
                        let x = (v[j] as f64 / 10000.0 / 0.18).max(0.0).powf(1.0 / 1.6).min(1.0);
                        rgba[4 * i + j] = (x * 255.0).round() as u8;
                    }
                    rgba[4 * i + 3] = 255;
                }
            }
        } else {
            use rayon::prelude::*;
            let needed: Vec<_> = needed.into_iter().collect();
            let blocks: Vec<_> = needed.par_iter().map(|&(k, bi, bj)| s.block(hits[k], year, bi, bj).map(|b| ((k, bi, bj), b))).collect::<Result<_>>()?;
            let blocks: std::collections::HashMap<_, _> = blocks.into_iter().collect();
            for (i, a) in at.iter().enumerate() {
                if let Some((k, c, r)) = *a {
                    let b = &blocks[&(k, r / CORE, c / CORE)];
                    let cls = b[(r % CORE) * CORE + c % CORE];
                    if cls == NODATA {
                        continue;
                    }
                    rgba[4 * i..4 * i + 3].copy_from_slice(&COLORS[cls as usize % 10]);
                    rgba[4 * i + 3] = 255;
                }
            }
        }
    }
    let mut png = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png, 256, 256);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(&rgba)?;
    }
    Ok(png)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_from_a_composite() {
        let mut data = vec![0u16; 20];
        data[16] = 3000; // summer NIR
        data[19] = 1000; // summer SWIR 2.2
        data[12] = 500; // summer red
        let comp = store::Composite { size: 1, data };
        assert_eq!(spectral_index(&comp, "nbr").unwrap(), vec![5000]);
        assert_eq!(spectral_index(&comp, "ndvi").unwrap(), vec![7143]);
        assert!(spectral_index(&comp, "evi").is_none());
        let empty = store::Composite { size: 1, data: vec![0; 20] };
        assert_eq!(spectral_index(&empty, "nbr").unwrap(), vec![i16::MIN]);
    }

    #[test]
    fn npy_header_is_parsed() {
        let dir = std::env::temp_dir().join(format!("canopy-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.npy");
        let header = "{'descr': '<u2', 'fortran_order': False, 'shape': (2, 3), }";
        let pad = 64 - (10 + header.len() + 1) % 64;
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let full = format!("{header}{}\n", " ".repeat(pad));
        bytes.extend((full.len() as u16).to_le_bytes());
        bytes.extend(full.as_bytes());
        for v in 0u16..6 {
            bytes.extend(v.to_le_bytes());
        }
        std::fs::write(&p, bytes).unwrap();
        let (shape, data) = store::read_npy_u16(&p).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5]);
        std::fs::remove_dir_all(dir).ok();
    }
}
