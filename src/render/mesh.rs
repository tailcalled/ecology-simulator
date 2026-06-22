//! Builds a renderable triangle mesh from the spherical grid.
//!
//! Each Voronoi cell becomes a fan of triangles `(center, ring[k], ring[k+1])`. Every vertex
//! of a cell carries that cell's index, so the shader can look up the cell's data value (e.g.
//! temperature) from a storage buffer and color the whole cell uniformly. Geometry is static
//! for the lifetime of the grid — only the per-cell data buffer changes each frame.

use bytemuck::{Pod, Zeroable};

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
}
