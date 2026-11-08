//! Coordinates: WGS84 longitude/latitude, UTM (transverse Mercator, Krüger's series to fourth
//! order in n, millimetre accuracy within a zone), and Web Mercator map tiles.

use std::f64::consts::PI;

const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const K0: f64 = 0.9996;
const E0: f64 = 500_000.0;

struct Series {
    a: f64, // rectifying radius
    alpha: [f64; 4],
    beta: [f64; 4],
    delta: [f64; 4],
    c: f64, // 2 sqrt(n) / (1 + n)
}

fn series() -> Series {
    let n = F / (2.0 - F);
    let (n2, n3, n4) = (n * n, n * n * n, n * n * n * n);
    Series {
        a: A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0),
        alpha: [
            n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0,
            13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0,
            61.0 * n3 / 240.0 - 103.0 * n4 / 140.0,
            49561.0 * n4 / 161280.0,
        ],
        beta: [
            n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0 - n4 / 360.0,
            n2 / 48.0 + n3 / 15.0 - 437.0 * n4 / 1440.0,
            17.0 * n3 / 480.0 - 37.0 * n4 / 840.0,
            4397.0 * n4 / 161280.0,
        ],
        delta: [
            2.0 * n - 2.0 * n2 / 3.0 - 2.0 * n3 + 116.0 * n4 / 45.0,
            7.0 * n2 / 3.0 - 8.0 * n3 / 5.0 - 227.0 * n4 / 45.0,
            56.0 * n3 / 15.0 - 136.0 * n4 / 35.0,
            4279.0 * n4 / 630.0,
        ],
        c: 2.0 * n.sqrt() / (1.0 + n),
    }
}

/// A UTM zone, from its EPSG code (326zz north, 327zz south).
#[derive(Clone, Copy, Debug)]
pub struct Utm {
    lon0: f64,
    n0: f64,
}

impl Utm {
    pub fn from_epsg(epsg: u32) -> Option<Utm> {
        let (zone, south) = match epsg {
            32601..=32660 => (epsg - 32600, false),
            32701..=32760 => (epsg - 32700, true),
            _ => return None,
        };
        Some(Utm { lon0: (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0, n0: if south { 10_000_000.0 } else { 0.0 } })
    }

    /// (lon, lat) in degrees -> (easting, northing) in metres.
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let s = series();
        let phi = lat.to_radians();
        let lam = (lon - self.lon0).to_radians();
        let t = (phi.sin().atanh() - s.c * (s.c * phi.sin()).atanh()).sinh();
        let xi = t.atan2(lam.cos());
        let eta = (lam.sin() / (1.0 + t * t).sqrt()).atanh();
        let (mut e, mut n) = (eta, xi);
        for j in 0..4 {
            let k = 2.0 * (j + 1) as f64;
            e += s.alpha[j] * (k * xi).cos() * (k * eta).sinh();
            n += s.alpha[j] * (k * xi).sin() * (k * eta).cosh();
        }
        (E0 + K0 * s.a * e, self.n0 + K0 * s.a * n)
    }

    /// (easting, northing) -> (lon, lat) in degrees.
    pub fn inverse(&self, e: f64, n: f64) -> (f64, f64) {
        let s = series();
        let xi = (n - self.n0) / (K0 * s.a);
        let eta = (e - E0) / (K0 * s.a);
        let (mut xp, mut ep) = (xi, eta);
        for j in 0..4 {
            let k = 2.0 * (j + 1) as f64;
            xp -= s.beta[j] * (k * xi).sin() * (k * eta).cosh();
            ep -= s.beta[j] * (k * xi).cos() * (k * eta).sinh();
        }
        let chi = (xp.sin() / ep.cosh()).asin();
        let mut phi = chi;
        for j in 0..4 {
            phi += s.delta[j] * (2.0 * (j + 1) as f64 * chi).sin();
        }
        let lam = ep.sinh().atan2(xp.cos());
        (self.lon0 + lam.to_degrees(), phi.to_degrees())
    }
}

/// Centre of pixel (px, py) of Web Mercator tile (z, x, y), 256 pixels a side -> (lon, lat).
pub fn tile_pixel(z: u32, x: u32, y: u32, px: f64, py: f64) -> (f64, f64) {
    let size = 256.0 * (1u64 << z) as f64;
    let gx = (x as f64 * 256.0 + px + 0.5) / size;
    let gy = (y as f64 * 256.0 + py + 0.5) / size;
    let lon = gx * 360.0 - 180.0;
    let lat = (PI * (1.0 - 2.0 * gy)).sinh().atan().to_degrees();
    (lon, lat)
}

/// Bounds of tile (z, x, y): (west, south, east, north) in degrees.
pub fn tile_bounds(z: u32, x: u32, y: u32) -> (f64, f64, f64, f64) {
    let (w, n) = tile_pixel(z, x, y, -0.5, -0.5);
    let (e, s) = tile_pixel(z, x, y, 255.5, 255.5);
    (w, s, e, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference points computed with PROJ (pyproj / rasterio): EPSG:4326 -> EPSG:326zz.
    const CASES: &[(u32, f64, f64, f64, f64)] = &[
        (32634, 23.64, 46.77, 701562.8654, 5182990.1555),
        (32635, 26.02, 44.94, 422680.4253, 4976752.3003),
        (32635, 29.50, 45.30, 696006.7347, 5019317.9653),
        (32634, 21.23, 45.75, 517889.4685, 5066297.1078),
        (32635, 24.10, 43.82, 266789.0680, 4855969.2155),
        (32634, 20.50, 47.90, 462630.2730, 5305306.7223),
        (32635, 30.10, 44.00, 748544.1411, 4876545.8953),
    ];

    #[test]
    fn utm_matches_proj() {
        for &(epsg, lon, lat, e, n) in CASES {
            let u = Utm::from_epsg(epsg).unwrap();
            let (ge, gn) = u.forward(lon, lat);
            assert!((ge - e).abs() < 0.001 && (gn - n).abs() < 0.001, "{epsg} {lon} {lat}: {ge} {gn} vs {e} {n}");
            let (glon, glat) = u.inverse(e, n);
            assert!((glon - lon).abs() < 1e-9 && (glat - lat).abs() < 1e-9, "inverse {glon} {glat}");
        }
    }

    #[test]
    fn tiles_cover_the_world() {
        let (w, s, e, n) = tile_bounds(0, 0, 0);
        assert!((w + 180.0).abs() < 1e-9 && (e - 180.0).abs() < 1e-9);
        assert!((n - 85.051_128_779_806_6).abs() < 1e-6 && (s + 85.051_128_779_806_6).abs() < 1e-6);
        // Cluj at zoom 12 lies in tile (2316, 1444).
        let (w, s, e, n) = tile_bounds(12, 2316, 1444);
        assert!(w < 23.6 && e > 23.6 && s < 46.77 && n > 46.77, "{w} {s} {e} {n}");
    }
}
