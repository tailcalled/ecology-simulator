//! Builds a renderable triangle mesh from the spherical grid.
//!
//! Each Voronoi cell becomes a fan of triangles `(center, ring[k], ring[k+1])`. Every vertex
//! of a cell carries that cell's index, so the shader can look up the cell's data value (e.g.
//! temperature) from a storage buffer and color the whole cell uniformly. Geometry is static
//! for the lifetime of the grid — only the per-cell data buffer changes each frame.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use crate::grid::{fibonacci_sphere, Grid};

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

/// Each cell's center as (longitude, latitude) in radians. The map-projection vertex shader uses
/// the center longitude to unwrap each cell's ring vertices onto a single branch, so cells that
/// straddle the ±180° antimeridian don't smear across the whole map.
pub fn build_cell_centers(grid: &Grid) -> Vec<Vec2> {
    grid.lonlat_deg
        .iter()
        .map(|ll| Vec2::new(ll.x.to_radians(), ll.y.to_radians()))
        .collect()
}

/// Slightly outside the unit sphere, so overlay lines sit just above the cell surface (and the
/// sphere still occludes the ones on the far side via the depth test).
const OVERLAY_RADIUS: f32 = 1.012;
/// World-space length of the longest motion arrow (great-circle radians ≈ chord at this scale).
const ARROW_LEN: f32 = 0.06;

/// Pick ~`target` cells spread *evenly over the sphere* to anchor an arrow field. Striding through
/// the cell index instead would walk the Fibonacci lattice's own spiral and draw arrows in visible
/// spiral lines; here we lay down `target` evenly-spaced directions (a coarse Fibonacci set) and
/// snap each to its nearest cell, deduping collisions. The result depends only on the grid, so it
/// is computed once and reused every frame.
pub fn arrow_sample_cells(grid: &Grid, target: usize) -> Vec<u32> {
    if grid.n == 0 || target == 0 {
        return Vec::new();
    }
    let mut seen = vec![false; grid.n];
    let mut out = Vec::with_capacity(target);
    for p in fibonacci_sphere(target.min(grid.n)) {
        let cell = grid.nearest_cell(p.as_vec3());
        if !seen[cell] {
            seen[cell] = true;
            out.push(cell as u32);
        }
    }
    out
}

/// Append one arrow (shaft + two barbs, 6 vertices) for tangent vector `v` at cell center `p` to
/// `out`. `vmax` sets the length scale (longest arrow). No-op for ~zero or non-tangent vectors.
fn emit_arrow(out: &mut Vec<Vec3>, p: Vec3, v: Vec3, vmax: f32) {
    let speed = v.length();
    if speed <= 1e-6 {
        return;
    }
    // Direction of motion in the cell's tangent plane.
    let dir = (v - p * v.dot(p)).normalize_or_zero();
    if dir.length_squared() <= 0.5 {
        return;
    }
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
    out.extend_from_slice(&[base, tip, tip, e1, tip, e2]);
}

/// Build a motion-arrow line list (vertex pairs) at a precomputed set of evenly-spread sample
/// cells (see [`arrow_sample_cells`]). Arrow length scales with each cell's speed relative to the
/// fastest *among the samples*. Empty when there is no field or no samples.
pub fn build_arrows_at(grid: &Grid, field: &[Vec3], samples: &[u32]) -> Vec<Vec3> {
    if field.is_empty() || samples.is_empty() {
        return Vec::new();
    }
    let vmax = samples
        .iter()
        .map(|&i| field[i as usize].length())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let mut out = Vec::with_capacity(samples.len() * 6);
    for &i in samples {
        emit_arrow(&mut out, grid.centers[i as usize], field[i as usize], vmax);
    }
    out
}

/// Convenience wrapper: build an arrow field sampling ~`target` evenly-spread cells. Recomputes the
/// sample set each call, so callers that draw every frame should instead cache
/// [`arrow_sample_cells`] and call [`build_arrows_at`].
pub fn build_arrows(grid: &Grid, velocity: &[Vec3], target: usize) -> Vec<Vec3> {
    if velocity.is_empty() || grid.n == 0 {
        return Vec::new();
    }
    let samples = arrow_sample_cells(grid, target);
    build_arrows_at(grid, velocity, &samples)
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
