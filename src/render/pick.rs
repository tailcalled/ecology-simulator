//! Screen-space picking: turn a normalized device coordinate (NDC) under the cursor into the
//! surface point it points at. Pure math (no wgpu), so it unit-tests on the host and is shared
//! by both views — only the camera's `view_proj` / `eye` differ.

use glam::{Mat4, Vec2, Vec3, Vec4};

/// Intersect the ray through NDC `ndc` (x, y each in [-1, 1], y up) with the unit sphere at the
/// origin, returning the (normalized) hit point nearest the camera, or `None` if the ray misses.
///
/// `view_proj` and `eye` come from the view's camera. The ray is reconstructed by unprojecting
/// the near/far NDC points through the inverse view-projection matrix.
pub fn ray_sphere(view_proj: Mat4, eye: Vec3, ndc: Vec2) -> Option<Vec3> {
    let inv = view_proj.inverse();
    let unproject = |z: f32| {
        let p = inv * Vec4::new(ndc.x, ndc.y, z, 1.0);
        p.truncate() / p.w
    };
    // WebGPU clip space has z in [0, 1]; sample both ends to get the ray direction.
    let near = unproject(0.0);
    let far = unproject(1.0);
    let dir = (far - near).normalize();

    // Solve |eye + t·dir|² = 1 for the smallest non-negative t.
    let b = dir.dot(eye);
    let c = eye.length_squared() - 1.0;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t_near = -b - sqrt_disc;
    let t = if t_near >= 0.0 { t_near } else { -b + sqrt_disc };
    if t < 0.0 {
        return None;
    }
    Some((eye + dir * t).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::camera::{OrbitCamera, SurfaceCamera};

    #[test]
    fn center_ray_hits_point_facing_camera() {
        let cam = OrbitCamera::default();
        let hit = ray_sphere(cam.view_proj(1.5), cam.eye(), Vec2::ZERO)
            .expect("center of screen should hit the sphere");
        // The point under the screen center is the one directly facing the eye.
        let facing = cam.eye().normalize();
        assert!(hit.dot(facing) > 0.999, "hit {hit:?} should face the camera");
    }

    #[test]
    fn corner_ray_misses() {
        let cam = OrbitCamera::default();
        // Far corner of the (wide) frustum points past the sphere's limb into empty space.
        assert!(ray_sphere(cam.view_proj(1.0), cam.eye(), Vec2::new(1.0, 1.0)).is_none());
    }

    #[test]
    fn surface_center_ray_hits_target() {
        let cam = SurfaceCamera::default();
        let hit = ray_sphere(cam.view_proj(1.0), cam.eye(), Vec2::ZERO)
            .expect("zoom center should hit the sphere");
        assert!(hit.dot(cam.target.normalize()) > 0.999);
    }
}
