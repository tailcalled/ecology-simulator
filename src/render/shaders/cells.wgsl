// Renders the spherical grid: each vertex carries its cell index, the fragment colors the
// cell by looking up that cell's data value (e.g. temperature) and mapping it through a
// colormap. Optional overlays: day/night sunlight shading and a lat/long graticule. A gentle
// view-dependent shade gives the sphere 3D form without obscuring data.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,      // world-space camera position (xyz)
    sun: vec4<f32>,      // direction to the sun in the planet frame (xyz)
    params: vec4<f32>,   // x = data min, y = data max, z = show sunlight, w = show graticule
    highlight: vec4<f32>,// x = hovered cell index (or -1), y = layer index (0 temp, 1 plates),
                         // z = projection (0 sphere, 1 Winkel Tripel map)
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> cell_data: array<f32>;
@group(2) @binding(0) var<storage, read> plate_data: array<u32>;
// Per-cell center (longitude, latitude) in radians; used to unwrap the seam in map projections.
@group(3) @binding(0) var<storage, read> cell_center: array<vec2<f32>>;

const PI: f32 = 3.141592653589793;

// Forward Winkel Tripel: (longitude, latitude) radians → map (x, y). Mirrors `winkel_tripel` in
// camera.rs (kept in sync so host-side picking can invert it).
fn winkel_xy(lon: f32, lat: f32) -> vec2<f32> {
    let cos_phi1 = 2.0 / PI;                 // standard parallel, cos φ₁ = 2/π
    let half_lon = 0.5 * lon;
    let alpha = acos(clamp(cos(lat) * cos(half_lon), -1.0, 1.0));
    var d = 1.0;                             // D = α / sin α → 1 as α → 0
    if (abs(alpha) > 1e-7) { d = alpha / sin(alpha); }
    let x = 0.5 * (lon * cos_phi1 + 2.0 * cos(lat) * sin(half_lon) * d);
    let y = 0.5 * (lat + sin(lat) * d);
    return vec2<f32>(x, y);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) value: f32,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) @interpolate(flat) cell: u32,
    @location(4) @interpolate(flat) plate: u32,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) cell: u32) -> VsOut {
    var out: VsOut;
    var world_pos = pos;
    if (camera.highlight.z > 0.5) {
        // Map projection: flatten the sphere onto the z = 0 plane.
        let n = normalize(pos);
        let lat = asin(clamp(n.z, -1.0, 1.0));
        var lon = atan2(n.y, n.x);
        // Unwrap onto the same 2π branch as this cell's center so cells spanning the ±180°
        // antimeridian stay on one side instead of stretching across the whole map.
        let clon = cell_center[cell].x;
        lon = lon - 2.0 * PI * round((lon - clon) / (2.0 * PI));
        let m = winkel_xy(lon, lat);
        world_pos = vec3<f32>(m.x, m.y, 0.0);
    }
    out.clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.value = cell_data[cell];
    out.world = pos;            // keep the sphere position for shading / graticule / sunlight
    out.normal = normalize(pos);
    out.cell = cell;
    out.plate = plate_data[cell];
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

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let h = fract(c.x) * 6.0;
    let s = c.y;
    let v = c.z;
    let f = h - floor(h);
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let m = i32(floor(h)) % 6;
    if (m == 0) { return vec3<f32>(v, t, p); }
    else if (m == 1) { return vec3<f32>(q, v, p); }
    else if (m == 2) { return vec3<f32>(p, v, t); }
    else if (m == 3) { return vec3<f32>(p, q, v); }
    else if (m == 4) { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

// Distinct categorical color per plate id. The golden-ratio hue step spreads adjacent ids far
// apart on the color wheel, and small id-keyed wobbles in saturation/value separate neighbors
// that happen to land near the same hue.
fn plate_color(id: u32) -> vec3<f32> {
    let fid = f32(id);
    let h = fract(fid * 0.61803398875);
    let s = 0.5 + 0.2 * fract(fid * 0.7);
    let v = 0.78 + 0.12 * fract(fid * 0.37);
    return hsv2rgb(vec3<f32>(h, s, v));
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
    // Layer 0 = continuous data through the colormap; layer 1 = categorical plate palette.
    var color: vec3<f32>;
    if (camera.highlight.y > 0.5) {
        color = plate_color(in.plate);
    } else {
        let lo = camera.params.x;
        let hi = camera.params.y;
        let t = (in.value - lo) / max(hi - lo, 1e-3);
        color = colormap(t);
    }

    let n = normalize(in.normal);

    // Subtle shading: brighter where the surface faces the camera, dimmer at the rim. The flat
    // map has no rim and no meaningful eye direction, so this only applies to the 3D globe.
    if (camera.highlight.z < 0.5) {
        let view_dir = normalize(camera.eye.xyz - in.world);
        let facing = clamp(dot(n, view_dir), 0.0, 1.0);
        color *= 0.55 + 0.45 * facing;
    }

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

    // Highlight the cell under the cursor: brighten and tint toward white.
    if (camera.highlight.x >= 0.0 && in.cell == u32(camera.highlight.x)) {
        color = mix(color, vec3<f32>(1.0, 1.0, 1.0), 0.4) * 1.25;
    }

    return vec4<f32>(color, 1.0);
}

// --- Footprint marker (zoom region outline drawn on the globe view) ---

@vertex
fn vs_marker(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    var world_pos = pos;
    if (camera.highlight.z > 0.5) {
        let n = normalize(pos);
        let lat = asin(clamp(n.z, -1.0, 1.0));
        let lon = atan2(n.y, n.x);
        let m = winkel_xy(lon, lat);
        // z > 0 lifts the line toward the eye (smaller clip depth than the z = 0 cells) so the
        // overlay isn't rejected by the depth test on the flat map.
        world_pos = vec3<f32>(m.x, m.y, 0.02);
    }
    return camera.view_proj * vec4<f32>(world_pos, 1.0);
}

@fragment
fn fs_marker() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.85, 0.2, 1.0); // amber outline
}

// --- Plate-motion arrows (line list, shares vs_marker; drawn just above the surface) ---

@fragment
fn fs_arrow() -> @location(0) vec4<f32> {
    return vec4<f32>(0.95, 0.97, 1.0, 1.0); // near-white
}
