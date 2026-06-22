//! Spherical Voronoi tessellation of a point set, via `d3_geo_voronoi_rs`.
//!
//! The library computes a Delaunay triangulation on the sphere (equivalently, the 3D convex
//! hull of the points). We take its raw, correctly-indexed outputs:
//!   * `triangles` — point-index triples,
//!   * `centers`   — one circumcenter per triangle (a Voronoi vertex), and
//!   * `neighbors` — per-point adjacency,
//! and build the per-cell Voronoi polygons ourselves as the dual (each cell = the
//! circumcenters of the triangles incident to that point).
//!
//! NOTE: we deliberately do **not** use the library's `delaunay.polygons`. That field is
//! collected by iterating an internal `HashMap` keyed by point index, but the key is dropped,
//! so its entries are neither aligned with point order nor deterministic across runs.
//!
//! Computation is done in f64 — the crate's README warns that f32 corrupts cell integrity.

use d3_geo_voronoi_rs::delaunay::Delaunay;
use geo_types::Coord;
use glam::DVec3;

/// Raw spherical Delaunay/Voronoi data, all correctly indexed by point or triangle order.
pub struct Tessellation {
    /// One Voronoi vertex (triangle circumcenter) per Delaunay triangle, as a unit vector.
    pub centers: Vec<DVec3>,
    /// Delaunay triangles as triples of point indices.
    pub triangles: Vec<[usize; 3]>,
    /// Per point: indices of adjacent points (cells sharing a Voronoi edge).
    pub neighbors: Vec<Vec<usize>>,
}

/// Convert a unit vector to (longitude, latitude) in **degrees** — the convention the geo
/// stack expects (cartesian = [cos·lat·cos·lon, cos·lat·sin·lon, sin·lat], z = polar axis).
pub fn to_lonlat_deg(p: DVec3) -> Coord<f64> {
    let lat = p.z.clamp(-1.0, 1.0).asin().to_degrees();
    let lon = p.y.atan2(p.x).to_degrees();
    Coord { x: lon, y: lat }
}

/// Inverse of [`to_lonlat_deg`]: (lon, lat) degrees → unit vector.
pub fn from_lonlat_deg(c: Coord<f64>) -> DVec3 {
    let lon = c.x.to_radians();
    let lat = c.y.to_radians();
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
}

/// Tessellate the sphere over `points` (assumed unit vectors).
pub fn tessellate(points: &[DVec3]) -> Tessellation {
    let coords: Vec<Coord<f64>> = points.iter().copied().map(to_lonlat_deg).collect();

    let delaunay = Delaunay::<f64>::try_from(&coords)
        .expect("Delaunay triangulation failed (need >= 4 non-degenerate points)");

    // One circumcenter per triangle; convert lon/lat degrees → unit vectors.
    let centers: Vec<DVec3> = delaunay
        .centers
        .iter()
        .take(delaunay.triangles.len()) // ignore any appended boundary "supplement" centers
        .map(|c| from_lonlat_deg(*c).normalize())
        .collect();

    let triangles: Vec<[usize; 3]> = delaunay.triangles.iter().copied().collect();

    // `delaunay.neighbors` is a HashMap<point, Vec<point>>; flatten to a dense Vec.
    let n = points.len();
    let mut neighbors = vec![Vec::new(); n];
    for (&cell, adj) in delaunay.neighbors.iter() {
        if cell < n {
            neighbors[cell] = adj.clone();
        }
    }

    Tessellation { centers, triangles, neighbors }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::fibonacci::fibonacci_sphere;

    #[test]
    fn lonlat_roundtrip() {
        for p in fibonacci_sphere(2000) {
            let back = from_lonlat_deg(to_lonlat_deg(p));
            assert!((p - back).length() < 1e-9, "roundtrip drift for {p:?}");
        }
    }

    #[test]
    fn tessellation_is_well_formed() {
        let pts = fibonacci_sphere(2000);
        let t = tessellate(&pts);
        assert_eq!(t.neighbors.len(), pts.len());
        // A closed sphere triangulation of N points has 2N-4 triangles.
        assert_eq!(t.triangles.len(), 2 * pts.len() - 4);
        assert_eq!(t.centers.len(), t.triangles.len());
        for (i, adj) in t.neighbors.iter().enumerate() {
            assert!(!adj.is_empty(), "cell {i} has no neighbors");
        }
        // Every triangle's circumcenter should be near its three corner points (a valid
        // Voronoi vertex sits inside the cells it borders, not on the far hemisphere).
        let centers: Vec<DVec3> = pts.iter().map(|p| p.normalize()).collect();
        for (tri, c) in t.triangles.iter().zip(&t.centers) {
            for &p in tri {
                assert!(c.dot(centers[p]) > 0.5, "circumcenter far from its triangle corner");
            }
        }
    }
}
