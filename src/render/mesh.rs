//! Builds a renderable triangle mesh from the spherical grid.
//!
//! Each Voronoi cell becomes a fan of triangles `(center, ring[k], ring[k+1])`. Every vertex
//! of a cell carries that cell's index, so the shader can look up the cell's data value (e.g.
//! temperature) from a storage buffer and color the whole cell uniformly. Geometry is static
//! for the lifetime of the grid — only the per-cell data buffer changes each frame.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::grid::Grid;

/// One mesh vertex: a position on the unit sphere plus the index of the cell it belongs to.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub cell: u32,
}

/// Expand the grid's Voronoi cells into a non-indexed triangle list.
pub fn build_mesh(grid: &Grid) -> Vec<Vertex> {
    let mut verts = Vec::new();
    for i in 0..grid.n {
        let center = grid.centers[i];
        let start = grid.ring_start[i] as usize;
        let end = grid.ring_start[i + 1] as usize;
        let ring = &grid.ring_verts[start..end];
        let m = ring.len();
        if m < 3 {
            continue;
        }
        let cell = i as u32;
        for k in 0..m {
            let a = ring[k];
            let b = ring[(k + 1) % m];
            verts.push(Vertex { pos: center.to_array(), cell });
            verts.push(Vertex { pos: a.to_array(), cell });
            verts.push(Vertex { pos: b.to_array(), cell });
        }
    }
    verts
}

/// Slightly outside the unit sphere, so overlay lines sit just above the cell surface (and the
/// sphere still occludes the ones on the far side via the depth test).
const OVERLAY_RADIUS: f32 = 1.012;
/// World-space length of the longest motion arrow (great-circle radians ≈ chord at this scale).
const ARROW_LEN: f32 = 0.06;

/// Build a sparse field of motion arrows as a line list (vertex pairs), sampling roughly
/// `target` cells. Each arrow is three segments — a shaft from the cell plus two barbs — drawn
/// just above the surface; arrow length scales with the cell's speed relative to the fastest.
/// Returns an empty vec when there is no velocity field (terrain not generated).
pub fn build_arrows(grid: &Grid, velocity: &[Vec3], target: usize) -> Vec<Vec3> {
    if velocity.is_empty() || grid.n == 0 {
        return Vec::new();
    }
    let n = grid.n;
    let stride = (n / target.max(1)).max(1);
    let vmax = velocity
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max)
        .max(1e-6);

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let p = grid.centers[i];
        let v = velocity[i];
        let speed = v.length();
        if speed > 1e-6 {
            // Direction of motion in the cell's tangent plane.
            let dir = (v - p * v.dot(p)).normalize_or_zero();
            if dir.length_squared() > 0.5 {
                let len = ARROW_LEN * (speed / vmax).clamp(0.25, 1.0);
                let base = p * OVERLAY_RADIUS;
                let tip = (p + dir * len).normalize();
                // Barbs: swept back from the tip by ~26° on either side, in the tangent plane.
                let side = p.cross(dir).normalize_or_zero();
                let back = -dir;
                let (s, cth) = 0.45f32.sin_cos();
                let barb_len = len * 0.4;
                let e1 = (tip + (back * cth + side * s) * barb_len).normalize() * OVERLAY_RADIUS;
                let e2 = (tip + (back * cth - side * s) * barb_len).normalize() * OVERLAY_RADIUS;
                let tip = tip * OVERLAY_RADIUS;
                out.push(base);
                out.push(tip);
                out.push(tip);
                out.push(e1);
                out.push(tip);
                out.push(e2);
            }
        }
        i += stride;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_is_three_verts_per_triangle() {
        let g = Grid::new(2000);
        let mesh = build_mesh(&g);
        assert_eq!(mesh.len() % 3, 0, "vertex count must be a multiple of 3");
        // Total triangles equals the total number of ring edges.
        let edges: usize = (0..g.n)
            .map(|i| (g.ring_start[i + 1] - g.ring_start[i]) as usize)
            .sum();
        assert_eq!(mesh.len(), edges * 3);
        // Every cell index is in range.
        assert!(mesh.iter().all(|v| (v.cell as usize) < g.n));
    }

    #[test]
    fn arrows_are_line_segments_above_the_surface() {
        let g = Grid::new(2000);
        // A uniform "velocity" field tangent at each cell (east-ish): cross with +Z.
        let vel: Vec<Vec3> = g.centers.iter().map(|&p| Vec3::Z.cross(p)).collect();
        let arrows = build_arrows(&g, &vel, 200);
        assert!(!arrows.is_empty());
        // Three segments per arrow → 6 vertices.
        assert_eq!(arrows.len() % 6, 0);
        // All vertices sit just above the unit sphere.
        assert!(arrows.iter().all(|p| (p.length() - OVERLAY_RADIUS).abs() < 1e-4));
        // No velocity field → no arrows.
        assert!(build_arrows(&g, &[], 200).is_empty());
    }
}
