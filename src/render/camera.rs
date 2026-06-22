//! Cameras for the two views. The planet is the unit sphere centered at the origin with +Z
//! as the north pole. Matrices use the WebGPU/wgpu clip convention (depth 0..1), so we use
//! glam's `*_rh` (not `*_rh_gl`) projections.

use glam::{Mat4, Vec3};

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
    }
}
