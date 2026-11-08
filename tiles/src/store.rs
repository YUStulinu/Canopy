//! Regions on disk (composites as .npy, grid in meta.json), and their class maps, computed block
//! by block on demand and cached.

use crate::geo::Utm;
use crate::infer::Model;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const CHANNELS: usize = 20;
/// Blocks: the network sees 256 x 256 pixels and keeps the centre 192 x 192.
pub const CORE: usize = 192;
pub const MARGIN: usize = 32;
pub const NODATA: u8 = 255;

#[derive(Serialize, Clone)]
pub struct RegionInfo {
    pub name: String,
    pub split: String,
    pub epsg: u32,
    /// Upper-left corner of the grid (metres), and its size in 10 m pixels.
    pub left: f64,
    pub top: f64,
    pub size: usize,
    pub years: Vec<i32>,
    /// Longitude/latitude of the four corners: upper-left, upper-right, lower-right, lower-left.
    pub corners: [[f64; 2]; 4],
    pub bbox: [f64; 4],
}

pub struct Region {
    pub info: RegionInfo,
    pub utm: Utm,
    dir: PathBuf,
}

impl Region {
    /// Pixel (column, row) of a longitude/latitude, if inside.
    pub fn pixel(&self, lon: f64, lat: f64) -> Option<(usize, usize)> {
        let (e, n) = self.utm.forward(lon, lat);
        let c = (e - self.info.left) / 10.0;
        let r = (self.info.top - n) / 10.0;
        let s = self.info.size as f64;
        if c >= 0.0 && r >= 0.0 && c < s && r < s {
            Some((c as usize, r as usize))
        } else {
            None
        }
    }
}

/// A composite: CHANNELS x size x size reflectances (x 10000), 0 where there is no data.
pub struct Composite {
    pub size: usize,
    pub data: Vec<u16>,
}

impl Composite {
    #[inline]
    pub fn at(&self, ch: usize, r: usize, c: usize) -> u16 {
        self.data[(ch * self.size + r) * self.size + c]
    }
}

pub fn read_npy_u16(path: &Path) -> Result<(Vec<usize>, Vec<u16>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("{} is not a .npy file", path.display());
    }
    let (hlen, start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen])?;
    if !header.contains("'descr': '<u2'") || header.contains("'fortran_order': True") {
        bail!("{}: expected little-endian uint16 in C order, got {header}", path.display());
    }
    let shape_txt = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).context("no shape")?;
    let shape: Vec<usize> = shape_txt.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let body = &bytes[start + hlen..];
    let count: usize = shape.iter().product();
    if body.len() < 2 * count {
        bail!("{}: truncated", path.display());
    }
    let data = body[..2 * count].chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    Ok((shape, data))
}

type Key = (String, i32);

pub struct Store {
    pub regions: Vec<Region>,
    pub model: Option<Model>,
    composites: Mutex<HashMap<Key, Arc<Composite>>>,
    blocks: Mutex<HashMap<(String, i32, usize, usize), Arc<Vec<u8>>>>,
    pub inference_ms: Mutex<Vec<f64>>,
}

impl Store {
    pub fn open(data: &Path, model: Option<Model>) -> Result<Store> {
        let mut regions = Vec::new();
        let mut dirs: Vec<_> = std::fs::read_dir(data)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("meta.json").exists()).collect();
        dirs.sort();
        for dir in dirs {
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
            let epsg = meta["epsg"].as_u64().context("epsg")? as u32;
            let t = meta["transform"].as_array().context("transform")?;
            let left = t[2].as_f64().context("transform")?;
            let top = t[5].as_f64().context("transform")?;
            let size = meta["size"].as_u64().context("size")? as usize;
            let mut years: Vec<i32> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".npy")).and_then(|n| n.parse().ok()))
                .collect();
            years.sort();
            let utm = Utm::from_epsg(epsg).context("not a UTM grid")?;
            let ext = size as f64 * 10.0;
            let corners = [(left, top), (left + ext, top), (left + ext, top - ext), (left, top - ext)].map(|(e, n)| {
                let (lon, lat) = utm.inverse(e, n);
                [lon, lat]
            });
            let lons = corners.map(|c| c[0]);
            let lats = corners.map(|c| c[1]);
            let bbox = [
                lons.iter().cloned().fold(f64::MAX, f64::min),
                lats.iter().cloned().fold(f64::MAX, f64::min),
                lons.iter().cloned().fold(f64::MIN, f64::max),
                lats.iter().cloned().fold(f64::MIN, f64::max),
            ];
            let name = meta["name"].as_str().map(String::from).unwrap_or_else(|| dir.file_name().unwrap().to_string_lossy().into());
            let split = meta["split"].as_str().unwrap_or("").to_string();
            regions.push(Region { info: RegionInfo { name, split, epsg, left, top, size, years, corners, bbox }, utm, dir });
        }
        Ok(Store { regions, model, composites: Mutex::default(), blocks: Mutex::default(), inference_ms: Mutex::default() })
    }

    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.info.name == name)
    }

    pub fn composite(&self, region: &Region, year: i32) -> Result<Arc<Composite>> {
        let key = (region.info.name.clone(), year);
        if let Some(c) = self.composites.lock().unwrap().get(&key) {
            return Ok(c.clone());
        }
        let (shape, data) = read_npy_u16(&region.dir.join(format!("{year}.npy")))?;
        if shape != [CHANNELS, region.info.size, region.info.size] {
            bail!("unexpected shape {shape:?}");
        }
        let c = Arc::new(Composite { size: region.info.size, data });
        self.composites.lock().unwrap().insert(key, c.clone());
        Ok(c)
    }

    /// Number of blocks along each side of a region.
    pub fn blocks_per_side(size: usize) -> usize {
        size.div_ceil(CORE)
    }

    /// The class map of one block (CORE x CORE, clipped at the region's edge).
    pub fn block(&self, region: &Region, year: i32, bi: usize, bj: usize) -> Result<Arc<Vec<u8>>> {
        let key = (region.info.name.clone(), year, bi, bj);
        if let Some(b) = self.blocks.lock().unwrap().get(&key) {
            return Ok(b.clone());
        }
        let model = self.model.as_ref().context("no model loaded")?;
        let comp = self.composite(region, year)?;
        let t = std::time::Instant::now();
        let classes = model.classify_block(&comp, bi * CORE, bj * CORE)?;
        self.inference_ms.lock().unwrap().push(t.elapsed().as_secs_f64() * 1000.0);
        let b = Arc::new(classes);
        self.blocks.lock().unwrap().insert(key, b.clone());
        Ok(b)
    }

    /// The whole class map of a region (size x size), every block computed (in parallel).
    pub fn class_map(&self, region: &Region, year: i32) -> Result<Vec<u8>> {
        use rayon::prelude::*;
        let n = region.info.size;
        let nb = Self::blocks_per_side(n);
        let jobs: Vec<(usize, usize)> = (0..nb).flat_map(|i| (0..nb).map(move |j| (i, j))).collect();
        let blocks: Vec<Arc<Vec<u8>>> = jobs.par_iter().map(|&(i, j)| self.block(region, year, i, j)).collect::<Result<_>>()?;
        let mut out = vec![NODATA; n * n];
        for (&(bi, bj), b) in jobs.iter().zip(&blocks) {
            for r in 0..CORE {
                let rr = bi * CORE + r;
                if rr >= n {
                    break;
                }
                for c in 0..CORE {
                    let cc = bj * CORE + c;
                    if cc >= n {
                        break;
                    }
                    out[rr * n + cc] = b[r * CORE + c];
                }
            }
        }
        Ok(out)
    }
}
