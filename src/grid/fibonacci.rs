//! Fibonacci (golden-angle) lattice: a near-uniform distribution of points on the unit
//! sphere. Cheap, deterministic, and dense-uniform — its only weakness (irregular local
//! adjacency) is resolved by computing a spherical Voronoi diagram over these points
//! (see [`super::voronoi`]).

use glam::DVec3;

/// The golden angle, π·(3 − √5) ≈ 2.399963 rad. Successive points are rotated by this much
/// in longitude, which is what makes the spiral fill the sphere evenly.
pub const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// Generate `n` near-uniformly distributed unit vectors on the sphere.
///
/// Points are placed on a spiral: the z (polar) coordinate steps uniformly from near +1 to
/// near −1 (equal-area in latitude bands), while longitude advances by the golden angle. The
/// `+0.5` offset centers each point in its latitude band, avoiding clustering at the poles.
///
/// `z` is the polar axis, so latitude = asin(z) and longitude = atan2(y, x) — matching the
/// (lon, lat) convention used by the Voronoi step.
pub fn fibonacci_sphere(n: usize) -> Vec<DVec3> {
    assert!(n >= 4, "need at least 4 points to tessellate a sphere");
    (0..n)
        .map(|i| {
            let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = GOLDEN_ANGLE * i as f64;
            let (s, c) = theta.sin_cos();
            DVec3::new(r * c, r * s, z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_points_are_unit_vectors() {
        for p in fibonacci_sphere(5000) {
            assert!((p.length() - 1.0).abs() < 1e-9, "point not on unit sphere: {p:?}");
        }
    }

    #[test]
    fn count_matches() {
        assert_eq!(fibonacci_sphere(1234).len(), 1234);
    }

    #[test]
    fn roughly_centered() {
        // A uniform distribution should have its centroid near the origin.
        let pts = fibonacci_sphere(20_000);
        let mean: DVec3 = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
        assert!(mean.length() < 1e-2, "centroid too far from origin: {}", mean.length());
    }
}
