//! Cameras for the two views. The planet is the unit sphere centered at the origin with +Z
//! as the north pole. Matrices use the WebGPU/wgpu clip convention (depth 0..1), so we use
//! glam's `*_rh` (not `*_rh_gl`) projections.

use glam::{Mat4, Vec3};

/// Winkel Tripel half-extents of the projected map (world units), used to fit it to the viewport.
/// Both fall out of the forward formula: x peaks at the equator's edge (λ = ±π, φ = 0) and y at
/// the poles (λ = 0, φ = ±π/2).
pub const WINKEL_HALF_W: f32 = 1.0 + std::f32::consts::FRAC_PI_2; // ≈ 2.5708
pub const WINKEL_HALF_H: f32 = std::f32::consts::FRAC_PI_2; // ≈ 1.5708
/// Standard-parallel cosine (Winkel's own choice, cos φ₁ = 2/π → φ₁ ≈ 50.46°).
const WINKEL_COS_PHI1: f32 = std::f32::consts::FRAC_2_PI;
/// Leave a small empty border so the map doesn't touch the canvas edges.
const WINKEL_MARGIN: f32 = 1.04;

/// Forward Winkel Tripel: (longitude, latitude) in radians → map (x, y) in world units. Mirrors
/// the GPU vertex-shader version in `cells.wgsl`; kept on the host for picking's inverse.
pub fn winkel_tripel(lon: f32, lat: f32) -> glam::Vec2 {
    let half_lon = 0.5 * lon;
    let alpha = (lat.cos() * half_lon.cos()).clamp(-1.0, 1.0).acos();
    // D = α / sin α (the reciprocal of the cardinal sine), → 1 as α → 0.
    let d = if alpha.abs() < 1e-7 { 1.0 } else { alpha / alpha.sin() };
    let x = 0.5 * (lon * WINKEL_COS_PHI1 + 2.0 * lat.cos() * half_lon.sin() * d);
    let y = 0.5 * (lat + lat.sin() * d);
    glam::Vec2::new(x, y)
}

/// Inverse Winkel Tripel: map (x, y) → (longitude, latitude) in radians, or `None` if the point
/// is off the map (or the iteration fails to converge). The projection has no closed-form
/// inverse, so this is Newton's method with a numerical Jacobian, seeded from the point itself.
pub fn winkel_tripel_inverse(x: f32, y: f32) -> Option<(f32, f32)> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let mut lon = x.clamp(-PI, PI);
    let mut lat = y.clamp(-FRAC_PI_2, FRAC_PI_2);
    let eps = 1e-4;
    for _ in 0..30 {
        let f = winkel_tripel(lon, lat);
        let dx = f.x - x;
        let dy = f.y - y;
        if dx * dx + dy * dy < 1e-12 {
            // Reject solutions pushed onto the clamp rails by an off-map target.
            if lon.abs() <= PI + 1e-3 && lat.abs() <= FRAC_PI_2 + 1e-3 {
                return Some((lon, lat));
            }
            return None;
        }
        // Numerical Jacobian via central differences.
        let fl = winkel_tripel(lon + eps, lat);
        let fr = winkel_tripel(lon - eps, lat);
        let fu = winkel_tripel(lon, lat + eps);
        let fd = winkel_tripel(lon, lat - eps);
        let j00 = (fl.x - fr.x) / (2.0 * eps);
        let j01 = (fu.x - fd.x) / (2.0 * eps);
        let j10 = (fl.y - fr.y) / (2.0 * eps);
        let j11 = (fu.y - fd.y) / (2.0 * eps);
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-9 {
            return None;
        }
        // Newton step: solve J · Δ = −residual, then clamp back into the valid domain.
        lon -= (j11 * dx - j01 * dy) / det;
        lat -= (-j10 * dx + j00 * dy) / det;
        lon = lon.clamp(-PI, PI);
        lat = lat.clamp(-FRAC_PI_2, FRAC_PI_2);
    }
    None
}

/// Orthographic view-projection that fits the whole Winkel Tripel map into a viewport of the
/// given aspect ratio (letterboxed, never stretched). The map is laid out in the z = 0 plane;
/// cells (built at z = 0) land at clip depth 0.5, leaving room for overlays lifted toward the eye.
pub fn winkel_view_proj(aspect: f32) -> Mat4 {
    let hw = WINKEL_HALF_W * WINKEL_MARGIN;
    let hh = WINKEL_HALF_H * WINKEL_MARGIN;
    let map_aspect = hw / hh;
    let aspect = aspect.max(1e-3);
    // Grow the half-extent of whichever axis has slack so the map keeps its true proportions.
    let (vw, vh) = if aspect >= map_aspect { (hh * aspect, hh) } else { (hw, hw / aspect) };
    Mat4::orthographic_rh(-vw, vw, -vh, vh, -1.0, 1.0)
}

/// Up vector that stays stable except exactly at the poles.
fn stable_up(dir: Vec3) -> Vec3 {
    // If we're looking nearly along Z, use Y as up to avoid a degenerate basis.
    if dir.z.abs() > 0.999 { Vec3::Y } else { Vec3::Z }
}

/// Orbit camera that frames the whole planet — the big-picture view.
///
/// `azimuth` rotates around the polar (Z) axis, `elevation` tilts up/down, `distance` is how
/// far the eye sits from the planet center (sphere radius = 1).
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { azimuth: 0.0, elevation: 0.35, distance: 2.6, fov_y: 45f32.to_radians() }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (se, ce) = self.elevation.sin_cos();
        let (sa, ca) = self.azimuth.sin_cos();
        self.distance * Vec3::new(ce * ca, ce * sa, se)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, stable_up(eye.normalize()));
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), 0.05, 10.0);
        proj * view
    }
}

/// Close-up camera aimed at a target point on the surface — the zoomed view.
///
/// `target` is a (not necessarily normalized) direction to the surface point of interest;
/// the eye sits just above it along its normal, looking down at the surface.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceCamera {
    pub target: Vec3,
    pub height: f32,
    pub fov_y: f32,
}

impl Default for SurfaceCamera {
    fn default() -> Self {
        // Tight, near-telephoto framing — the footprint half-angle ≈ height·tan(fov/2) ≈ 0.03
        // rad (~1.8°), roughly 8× more zoomed-in than a wide 35°/0.8 view.
        Self { target: Vec3::X, height: 0.3, fov_y: 12f32.to_radians() }
    }
}

impl SurfaceCamera {
    pub fn eye(&self) -> Vec3 {
        self.target.normalize() * (1.0 + self.height)
    }

    /// Orthonormal surface frame at the target: (outward normal, east, north).
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let n = self.target.normalize();
        let east = Vec3::Z.cross(n);
        let east = if east.length_squared() < 1e-8 { Vec3::X } else { east.normalize() };
        let north = n.cross(east);
        (n, east, north)
    }

    /// Move the target across the surface by small angles east / north (radians-ish).
    pub fn pan(&mut self, d_east: f32, d_north: f32) {
        let (n, east, north) = self.basis();
        self.target = (n + east * d_east + north * d_north).normalize();
    }

    /// Half-extent of the view's footprint on the surface, as a tangent-plane offset.
    fn footprint_extent(&self) -> f32 {
        self.height * (self.fov_y * 0.5).tan()
    }

    /// A closed line loop (world positions, lifted slightly off the surface) outlining the
    /// region this camera frames — drawn on the globe view to show where the zoom is looking.
    /// `segments` is the number of samples per side (edges curve to follow the sphere).
    pub fn footprint_outline(&self, segments: usize) -> Vec<Vec3> {
        let (n, east, north) = self.basis();
        let s = self.footprint_extent();
        let lift = 1.004;
        let corner = |sx: f32, sy: f32| (n + east * (s * sx) + north * (s * sy)).normalize();
        let corners = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
        let mut pts = Vec::with_capacity(segments * 4 + 1);
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            for k in 0..segments {
                let t = k as f32 / segments as f32;
                pts.push(a.lerp(b, t).normalize() * lift);
            }
        }
        pts.push(corners[0] * lift);
        pts
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let surf = self.target.normalize();
        let eye = surf * (1.0 + self.height);
        let view = Mat4::look_at_rh(eye, surf, stable_up(surf));
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), 0.01, 10.0);
        proj * view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_eye_is_at_distance() {
        let c = OrbitCamera::default();
        assert!((c.eye().length() - c.distance).abs() < 1e-5);
    }

    #[test]
    fn matrices_are_finite() {
        let vp = OrbitCamera::default().view_proj(1.5);
        assert!(vp.to_cols_array().iter().all(|x| x.is_finite()));
        let vp2 = SurfaceCamera::default().view_proj(1.5);
        assert!(vp2.to_cols_array().iter().all(|x| x.is_finite()));
        assert!(winkel_view_proj(1.6).to_cols_array().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn winkel_inverse_roundtrips() {
        use std::f32::consts::{FRAC_PI_2, PI};
        // Sweep a grid of lon/lat (just inside the poles/antimeridian) and check forward→inverse
        // returns the original coordinate.
        for i in -8..=8 {
            for j in -4..=4 {
                let lon = i as f32 / 9.0 * PI;
                let lat = j as f32 / 5.0 * FRAC_PI_2;
                let p = winkel_tripel(lon, lat);
                let (lon2, lat2) = winkel_tripel_inverse(p.x, p.y)
                    .unwrap_or_else(|| panic!("no inverse for lon={lon} lat={lat}"));
                assert!((lon - lon2).abs() < 1e-3, "lon {lon} vs {lon2}");
                assert!((lat - lat2).abs() < 1e-3, "lat {lat} vs {lat2}");
            }
        }
        // A point well outside the map's bounding box has no inverse.
        assert!(winkel_tripel_inverse(WINKEL_HALF_W + 1.0, 0.0).is_none());
    }
}
