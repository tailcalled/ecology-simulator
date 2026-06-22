//! Continental plates: a partition of the grid into rigid tectonic plates, each carrying a
//! direction of motion. This is the first geological layer; later it will feed elevation and
//! couple back into the climate (ocean currents, orographic effects), so it lives alongside the
//! climate state in [`Sim`](super::Sim) rather than in a wholly separate world.
//!
//! Generation is deterministic from a seed. Plates are grown by a randomized flood fill over the
//! grid adjacency graph — dropping a seed cell per plate, then repeatedly claiming a random
//! frontier cell for its plate — which yields organic, interlocking plate outlines rather than
//! the convex cells a nearest-seed Voronoi assignment would give.
//!
//! Each plate moves as a rigid rotation of the sphere about its own **Euler pole** (an axis
//! through the planet's center), which is the physically correct way a rigid cap moves on a
//! sphere. A cell's surface velocity is then `ω × p`, automatically tangent to the surface and
//! vanishing at the pole — so a single plate's arrows fan out and swirl around its pole.

use glam::Vec3;

use crate::grid::Grid;

/// One rigid plate: a rotation of the sphere about `axis` (a unit Euler pole through the center)
/// at signed angular speed `rate` (radians per simulation second; sign sets the spin direction).
#[derive(Clone, Copy, Debug)]
pub struct Plate {
    /// Euler pole: unit axis the plate rotates about.
    pub axis: Vec3,
    /// Signed angular speed about `axis` (rad·s⁻¹).
    pub rate: f32,
}

impl Plate {
    /// Surface velocity this plate imparts at unit position `p` (tangent to the sphere).
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        (self.axis * self.rate).cross(p)
    }
}

/// The tectonic partition of a [`Grid`]: which plate each cell belongs to, the plates' motions,
/// and the per-cell surface velocity those motions produce.
pub struct Terrain {
    /// Plate index per cell (parallel to the grid's cells).
    pub plate_id: Vec<u32>,
    /// The plates, indexed by the ids in `plate_id`.
    pub plates: Vec<Plate>,
    /// Per-cell tectonic surface velocity (`ω × p`), tangent to the sphere.
    pub velocity: Vec<Vec3>,
}

impl Terrain {
    /// An empty terrain (no plates), used before generation.
    pub fn empty() -> Self {
        Self { plate_id: Vec::new(), plates: Vec::new(), velocity: Vec::new() }
    }

    /// Partition `grid` into `num_plates` plates by randomized flood fill, assign each plate a
    /// random Euler pole and speed, and precompute the per-cell surface velocity. Deterministic
    /// in `seed`.
    pub fn generate(grid: &Grid, num_plates: usize, seed: u64) -> Self {
        let n = grid.n;
        let num_plates = num_plates.clamp(1, n.max(1));
        let mut rng = Rng::new(seed);

        // --- Pick distinct seed cells, one per plate. ---
        const UNASSIGNED: u32 = u32::MAX;
        let mut plate_id = vec![UNASSIGNED; n];
        let mut frontier: Vec<(u32, u32)> = Vec::new(); // (candidate cell, claiming plate)
        let mut planted = 0u32;
        while (planted as usize) < num_plates {
            let c = (rng.next_u64() % n as u64) as usize;
            if plate_id[c] == UNASSIGNED {
                plate_id[c] = planted;
                for &nb in grid.neighbors(c) {
                    frontier.push((nb, planted));
                }
                planted += 1;
            }
        }

        // --- Grow plates: repeatedly claim a random frontier cell for its plate. Picking the
        // frontier entry at random (rather than FIFO) breaks up the regular wavefront into
        // ragged, natural-looking boundaries. ---
        while !frontier.is_empty() {
            let k = (rng.next_u64() % frontier.len() as u64) as usize;
            let (cell, plate) = frontier.swap_remove(k);
            let cell = cell as usize;
            if plate_id[cell] != UNASSIGNED {
                continue;
            }
            plate_id[cell] = plate;
            for &nb in grid.neighbors(cell) {
                if plate_id[nb as usize] == UNASSIGNED {
                    frontier.push((nb, plate));
                }
            }
        }
        // The grid is connected, so every cell is claimed; guard the rare stray anyway.
        for id in plate_id.iter_mut() {
            if *id == UNASSIGNED {
                *id = 0;
            }
        }

        // --- Give each plate a random Euler pole and a signed angular speed. ---
        let plates: Vec<Plate> = (0..num_plates)
            .map(|_| {
                let axis = rng.unit_vec();
                // Magnitude in [0.3, 1.0] of the base rate, with a random spin direction. The
                // absolute value is arbitrary for now (only relative arrow lengths matter); it is
                // expressed as rad·s⁻¹ so it can drive real plate displacement later.
                let mag = 0.3 + 0.7 * rng.unit_f32();
                let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                Plate { axis, rate: BASE_RATE * mag * sign }
            })
            .collect();

        // --- Precompute the per-cell surface velocity from its plate's motion. ---
        let velocity: Vec<Vec3> = (0..n)
            .map(|i| plates[plate_id[i] as usize].velocity_at(grid.centers[i]))
            .collect();

        Self { plate_id, plates, velocity }
    }
}

/// Base plate angular speed (rad·s⁻¹). Earth's fast plates drift ~10 cm·yr⁻¹ ≈ 5e-17 rad·s⁻¹;
/// this stand-in is far larger because nothing integrates it yet — it only sets arrow scale.
const BASE_RATE: f32 = 1.0;

/// Tiny deterministic PRNG (SplitMix64). Self-contained so plate generation is reproducible
/// without pulling in a random-number crate, matching the grid's deterministic construction.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    fn unit_f32(&mut self) -> f32 {
        // Top 24 bits → exact float in [0, 1).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniformly-distributed unit vector on the sphere (z uniform, azimuth uniform).
    fn unit_vec(&mut self) -> Vec3 {
        let z = self.unit_f32() * 2.0 - 1.0;
        let theta = self.unit_f32() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cell_is_assigned_to_a_valid_plate() {
        let g = Grid::new(2000);
        let t = Terrain::generate(&g, 10, 42);
        assert_eq!(t.plate_id.len(), g.n);
        assert_eq!(t.plates.len(), 10);
        assert!(t.plate_id.iter().all(|&id| (id as usize) < t.plates.len()));
        // Every plate actually claimed at least its seed cell.
        for p in 0..t.plates.len() as u32 {
            assert!(t.plate_id.contains(&p), "plate {p} claimed no cells");
        }
    }

    #[test]
    fn velocity_is_tangent_to_the_surface() {
        let g = Grid::new(2000);
        let t = Terrain::generate(&g, 8, 7);
        for i in 0..g.n {
            // ω × p is perpendicular to p, so the radial component must be ~0.
            assert!(t.velocity[i].dot(g.centers[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let g = Grid::new(1500);
        let a = Terrain::generate(&g, 12, 12345);
        let b = Terrain::generate(&g, 12, 12345);
        assert_eq!(a.plate_id, b.plate_id);
        // Different seed gives a different partition.
        let c = Terrain::generate(&g, 12, 99999);
        assert_ne!(a.plate_id, c.plate_id);
    }
}
