// Renders the spherical grid: each vertex carries its cell index, the fragment colors the
// cell by looking up that cell's data value (e.g. temperature) and mapping it through a
// colormap. Optional overlays: day/night sunlight shading and a lat/long graticule. A gentle
// view-dependent shade gives the sphere 3D form without obscuring data.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,      // world-space camera position (xyz)
    sun: vec4<f32>,      // direction to the sun in the planet frame (xyz)
    params: vec4<f32>,   // x = data min, y = data max, z = show sunlight, w = show graticule
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> cell_data: array<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) value: f32,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) cell: u32) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.value = cell_data[cell];
    out.world = pos;
    out.normal = normalize(pos);
    return out;
}

// Blue -> cyan -> green -> yellow -> red ramp over t in [0, 1].
fn colormap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.03, 0.10, 0.45); // cold (deep blue)
    let c1 = vec3<f32>(0.10, 0.60, 0.85); // cyan
    let c2 = vec3<f32>(0.15, 0.70, 0.30); // green
    let c3 = vec3<f32>(0.95, 0.85, 0.20); // yellow
    let c4 = vec3<f32>(0.85, 0.20, 0.12); // hot (red)
    if (x < 0.25) {
        return mix(c0, c1, x / 0.25);
    } else if (x < 0.5) {
        return mix(c1, c2, (x - 0.25) / 0.25);
    } else if (x < 0.75) {
        return mix(c2, c3, (x - 0.5) / 0.25);
    }
    return mix(c3, c4, (x - 0.75) / 0.25);
}

// Anti-aliased graticule line intensity: 1 near a multiple of `spacing` degrees, else 0.
fn grid_line(value_deg: f32, spacing: f32) -> f32 {
    let scaled = value_deg / spacing;
    let dist = abs(fract(scaled + 0.5) - 0.5);      // fractional distance to nearest line
    let aa = fwidth(scaled) * 1.5;                   // ~1.5px wide in screen space
    return 1.0 - smoothstep(0.0, aa, dist);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let lo = camera.params.x;
    let hi = camera.params.y;
    let t = (in.value - lo) / max(hi - lo, 1e-3);
    var color = colormap(t);

    let n = normalize(in.normal);

    // Subtle shading: brighter where the surface faces the camera, dimmer at the rim.
    let view_dir = normalize(camera.eye.xyz - in.world);
    let facing = clamp(dot(n, view_dir), 0.0, 1.0);
    color *= 0.55 + 0.45 * facing;

    // Sunlight overlay: darken the night side, leaving a clear day/night terminator.
    if (camera.params.z > 0.5) {
        let sun_factor = max(dot(n, normalize(camera.sun.xyz)), 0.0);
        color *= 0.2 + 0.8 * sun_factor;
    }

    // Lat/long graticule overlay every 15 degrees.
    if (camera.params.w > 0.5) {
        let lat = degrees(asin(clamp(n.z, -1.0, 1.0)));
        let lon = degrees(atan2(n.y, n.x));
        let line = max(grid_line(lat, 15.0), grid_line(lon, 15.0));
        color = mix(color, vec3<f32>(0.92, 0.94, 1.0), line * 0.7);
    }

    return vec4<f32>(color, 1.0);
}

// --- Footprint marker (zoom region outline drawn on the globe view) ---

@vertex
fn vs_marker(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_marker() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.85, 0.2, 1.0); // amber outline
}
