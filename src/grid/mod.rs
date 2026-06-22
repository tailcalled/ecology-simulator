//! Near-uniform spherical grid: Fibonacci-lattice cell centers tessellated by a spherical
//! Voronoi diagram. This is the substrate every simulation layer runs on.
//!
//! Storage is flat / CSR (compressed-sparse-row) so the per-cell kernels are cache-friendly
//! and trivially parallelizable with rayon. The struct is deliberately free of `wgpu` and
//! `wasm-bindgen` so it builds and unit-tests on the host.

mod fibonacci;
mod voronoi;

pub use fibonacci::fibonacci_sphere;

use glam::{Vec2, Vec3};
use voronoi::{tessellate, to_lonlat_deg};

/// A spherical grid of `n` cells.
pub struct Grid {
    /// Number of cells.
    pub n: usize,
    /// Cell-center unit vectors (also the outward surface normals).
    pub centers: Vec<Vec3>,
    /// Cell centers as (longitude, latitude) in degrees.
    pub lonlat_deg: Vec<Vec2>,
    /// Cell areas in steradians; sums to ≈ 4π over the whole sphere.
    pub areas: Vec<f32>,

    // --- CSR adjacency: neighbors of cell `i` are adj[adj_start[i]..adj_start[i+1]] ---
    /// Offsets into `adj` / `adj_weight`, length `n + 1`.
    pub adj_start: Vec<u32>,
    /// Flattened neighbor cell indices.
    pub adj: Vec<u32>,
    /// Per-edge diffusion conductance (shared-edge arc length / center-to-center arc length).
    pub adj_weight: Vec<f32>,

    // --- CSR Voronoi cell polygons (for rendering): ring of cell `i` is
    //     ring_verts[ring_start[i]..ring_start[i+1]] ---
    /// Offsets into `ring_verts`, length `n + 1`.
    pub ring_start: Vec<u32>,
    /// Flattened cell-polygon vertices (unit vectors), ordered around each cell.
    pub ring_verts: Vec<Vec3>,
}

/// Great-circle angle between two unit vectors (robust near 0 and π).
fn arc(a: Vec3, b: Vec3) -> f32 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Signed area of the spherical triangle (a, b, c) via the Van Oosterom–Strackee formula.
/// Stable for the small triangles that make up a cell fan.
fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let triple = a.dot(b.cross(c));
    let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * triple.atan2(den)
}

impl Grid {
    /// Build a grid of `n` near-uniform cells.
    pub fn new(n: usize) -> Self {
        let points = fibonacci_sphere(n);
        let tess = tessellate(&points);

        let centers: Vec<Vec3> = points.iter().map(|p| p.as_vec3().normalize()).collect();
        let lonlat_deg: Vec<Vec2> = points
            .iter()
            .map(|p| {
                let c = to_lonlat_deg(*p);
                Vec2::new(c.x as f32, c.y as f32)
            })
            .collect();
        let voronoi_verts: Vec<Vec3> =
            tess.centers.iter().map(|v| v.as_vec3().normalize()).collect();

        // Dual construction: each cell's Voronoi polygon is the set of circumcenters of the
        // Delaunay triangles incident to that point.
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (t, tri) in tess.triangles.iter().enumerate() {
            for &p in tri {
                incident[p].push(t as u32);
            }
        }

        // --- Render rings (CSR) + cell areas ---
        // Sort each cell's vertices by azimuth around the cell normal so the fan triangulation
        // has consistent winding and the area is correct (and deterministic).
        let mut ring_start = Vec::with_capacity(n + 1);
        let mut ring_verts: Vec<Vec3> = Vec::new();
        let mut areas = vec![0.0f32; n];
        ring_start.push(0u32);
        for i in 0..n {
            let nrm = centers[i];
            // Build a tangent basis (u, v) in the plane perpendicular to the normal.
            let seed = if nrm.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let u = seed.cross(nrm).normalize();
            let v = nrm.cross(u);

            let mut pts: Vec<Vec3> =
                incident[i].iter().map(|&t| voronoi_verts[t as usize]).collect();
            pts.sort_by(|a, b| {
                let aa = a.dot(v).atan2(a.dot(u));
                let bb = b.dot(v).atan2(b.dot(u));
                aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let m = pts.len();
            let mut sum = 0.0;
            for k in 0..m {
                sum += spherical_triangle_area(nrm, pts[k], pts[(k + 1) % m]);
            }
            areas[i] = sum.abs();

            ring_verts.extend_from_slice(&pts);
            ring_start.push(ring_verts.len() as u32);
        }

        // --- Adjacency (CSR) with per-edge conductance weights ---
        let mut adj_start = Vec::with_capacity(n + 1);
        let mut adj = Vec::new();
        let mut adj_weight = Vec::new();
        adj_start.push(0u32);
        for i in 0..n {
            for &j in &tess.neighbors[i] {
                if j == i {
                    continue;
                }
                let dist = arc(centers[i], centers[j]).max(1e-6);
                // Shared Voronoi edge = circumcenters of the (usually 2) Delaunay triangles
                // incident to both points i and j.
                let shared: Vec<u32> = incident[i]
                    .iter()
                    .copied()
                    .filter(|t| incident[j].contains(t))
                    .collect();
                let weight = if shared.len() == 2 {
                    let edge = arc(
                        voronoi_verts[shared[0] as usize],
                        voronoi_verts[shared[1] as usize],
                    );
                    edge / dist
                } else {
                    // Degenerate adjacency (rare on a near-uniform grid): isotropic fallback.
                    1.0 / dist
                };
                adj.push(j as u32);
                adj_weight.push(weight);
            }
            adj_start.push(adj.len() as u32);
        }

        Grid { n, centers, lonlat_deg, areas, adj_start, adj, adj_weight, ring_start, ring_verts }
    }

    /// Neighbor indices of cell `i`.
    pub fn neighbors(&self, i: usize) -> &[u32] {
        &self.adj[self.adj_start[i] as usize..self.adj_start[i + 1] as usize]
    }

    /// Per-edge conductance weights for cell `i` (parallel to [`Self::neighbors`]).
    pub fn neighbor_weights(&self, i: usize) -> &[f32] {
        &self.adj_weight[self.adj_start[i] as usize..self.adj_start[i + 1] as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn areas_sum_to_four_pi() {
        let g = Grid::new(8000);
        let total: f64 = g.areas.iter().map(|&a| a as f64).sum();
        let expected = 4.0 * std::f64::consts::PI;
        let rel_err = (total - expected).abs() / expected;
        assert!(rel_err < 1e-3, "area sum {total} vs 4π {expected} (rel err {rel_err})");
    }

    #[test]
    fn adjacency_is_symmetric() {
        let g = Grid::new(4000);
        for i in 0..g.n {
            for &j in g.neighbors(i) {
                let j = j as usize;
                assert!(
                    g.neighbors(j).contains(&(i as u32)),
                    "edge {i}->{j} not mirrored {j}->{i}",
                );
            }
        }
    }

    #[test]
    fn neighbor_counts_are_reasonable() {
        // Near-uniform spherical tessellations have ~5–8 neighbors per cell.
        let g = Grid::new(4000);
        let mut min = usize::MAX;
        let mut max = 0;
        for i in 0..g.n {
            let d = g.neighbors(i).len();
            min = min.min(d);
            max = max.max(d);
        }
        assert!(min >= 4, "min degree {min} too low");
        assert!(max <= 10, "max degree {max} too high");
    }

    #[test]
    fn csr_offsets_are_monotonic() {
        let g = Grid::new(2000);
        assert_eq!(g.adj_start.len(), g.n + 1);
        assert_eq!(g.ring_start.len(), g.n + 1);
        assert!(g.adj_start.windows(2).all(|w| w[0] <= w[1]));
        assert!(g.ring_start.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(*g.adj_start.last().unwrap() as usize, g.adj.len());
        assert_eq!(*g.ring_start.last().unwrap() as usize, g.ring_verts.len());
    }
}
