//! A tiny deterministic PRNG shared across the simulation's generators.
//!
//! Everything in the world (the grid, the plates, the elevation field) is built deterministically
//! from a seed so a run is fully reproducible. This [`Rng`] is the common source of randomness for
//! the procedural-generation passes; it is self-contained (SplitMix64) so we don't pull in a
//! random-number crate, and so the exact stream is stable across platforms and toolchains.
//!
//! It is intentionally *not* cryptographic and not meant for statistical-quality Monte Carlo — it
//! just needs to be fast, well-distributed enough for worldgen, and perfectly reproducible.

use glam::Vec3;

/// Deterministic SplitMix64 generator. Seed it once; draw a reproducible stream.
pub(crate) struct Rng(u64);

impl Rng {
    /// Create a generator from a seed.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next raw 64-bit value (SplitMix64).
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    pub(crate) fn unit_f32(&mut self) -> f32 {
        // Top 24 bits → exact float in [0, 1).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard-normal sample via the Box–Muller transform.
    pub(crate) fn normal(&mut self) -> f32 {
        let u1 = self.unit_f32().max(1e-6);
        let u2 = self.unit_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniformly-distributed unit vector on the sphere (z uniform, azimuth uniform).
    pub(crate) fn unit_vec(&mut self) -> Vec3 {
        let z = self.unit_f32() * 2.0 - 1.0;
        let theta = self.unit_f32() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }
}
