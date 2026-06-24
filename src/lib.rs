//! WASM-powered, multithreaded ecology simulator.
//!
//! Architecture (forced by `wgpu` being `!Send`/`!Sync` once wasm `+atomics` is on):
//! the whole engine runs inside ONE dedicated Web Worker that owns both canvases as
//! `OffscreenCanvas`. `wasm-bindgen-rayon` spawns *additional* workers used only for CPU
//! compute. The DOM main thread does setup, UI, and per-frame `requestAnimationFrame` ticks.
//!
//! The engine (grid + simulation + renderer) lives in the worker; this file exposes the thin
//! `#[wasm_bindgen]` surface the worker JS calls into, plus the rayon pool initializer.

// Cross-platform simulation core lives in these modules so they can be `cargo test`ed on the
// host. They are intentionally free of wgpu / wasm-bindgen.
pub mod grid;
pub mod render;
pub(crate) mod rng;
pub mod sim;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;
    use web_sys::OffscreenCanvas;

    use glam::Vec2;

    use crate::grid::Grid;
    use crate::render::camera::{winkel_view_proj, OrbitCamera, SurfaceCamera};
    use crate::render::mesh::{
        arrow_sample_cells, build_arrows_at, build_cell_centers, build_mesh,
    };
    use crate::render::pick::{ray_sphere, unproject_winkel};
    use crate::render::{Layer, Projection, Renderer, ViewCamera};
    use crate::sim::{Climate, Sim};

    // Re-export the rayon thread-pool initializer; wasm-bindgen exposes it to JS as
    // `initThreadPool`. JS must call it (after `init()`) before any rayon work runs.
    #[allow(unused_imports)]
    pub use wasm_bindgen_rayon::init_thread_pool;

    /// Number of grid cells. ~16k gives good resolution while keeping the one-time Voronoi
    /// precompute and the mesh size modest.
    const GRID_CELLS: usize = 16_384;
    /// Initial fast-forward factor (simulation seconds advanced per real second). The UI speed
    /// slider overrides this; at 5000× one Earth day (86 400 s) passes in ~17 s.
    const DEFAULT_TIME_SCALE: f32 = 5_000.0;
    /// Big-picture view auto-rotation (rad/s). Cosmetic spin of the globe view only — it does
    /// not affect the simulation, whose day/night comes from moving the subsolar point.
    const AUTO_ROTATE: f32 = 0.08;
    /// Starting temperature for every cell — Earth's mean surface temperature (K).
    const INITIAL_TEMP: f32 = 288.0;
    /// Number of continental plates the terrain generator carves the planet into. Matches the
    /// 52-plate PB2002 catalogue (Bird 2003); the default plate-size model is tuned to this count.
    const NUM_PLATES: usize = 52;
    /// Seed for deterministic plate generation.
    const PLATE_SEED: u64 = 0xC0FFEE;
    /// Roughly how many cells to sample for the plate-motion arrow field.
    const ARROW_SAMPLES: usize = 600;

    /// The whole engine: grid + simulation + renderer + the two cameras. Lives in a worker
    /// thread-local because the renderer's GPU objects cannot cross threads.
    struct Engine {
        grid: Grid,
        sim: Sim,
        renderer: Renderer,
        globe: OrbitCamera,
        zoom: SurfaceCamera,
        /// Screen projection per view (index 0 = globe, 1 = zoomed). Defaults to the 3D globe.
        projection: [Projection; 2],
        /// Simulation seconds advanced per real second. 0 = paused, 1 = real-time.
        time_scale: f32,
        /// Evenly-spread cell indices anchoring the arrow overlays (plate motion, wind). Depends
        /// only on the grid, so it is computed once and reused every frame.
        arrow_samples: Vec<u32>,
        frame: u64,
    }

    thread_local! {
        static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    }

    /// Called once per thread when the wasm module is instantiated (including rayon workers).
    #[wasm_bindgen(start)]
    pub fn on_load() {
        console_error_panic_hook::set_once();
        // `console_log::init*` may be called once per thread; ignore the already-init error.
        let _ = console_log::init_with_level(log::Level::Info);
    }

    /// Build the engine from the two transferred OffscreenCanvases. Async because GPU adapter
    /// + device acquisition are async on the web. Stores the engine in the worker's TLS.
    #[wasm_bindgen]
    pub async fn engine_init(
        canvas0: OffscreenCanvas,
        canvas1: OffscreenCanvas,
    ) -> Result<(), JsValue> {
        let threads = rayon::current_num_threads();
        log::info!("engine_init: rayon thread pool has {threads} threads");

        let grid = Grid::new(GRID_CELLS);
        log::info!("engine_init: grid built with {} cells", grid.n);
        let mesh = build_mesh(&grid);
        log::info!("engine_init: mesh has {} vertices", mesh.len());

        let mut sim = Sim::new(grid.n, Climate::default(), INITIAL_TEMP);
        sim.generate_terrain(&grid, NUM_PLATES, PLATE_SEED);
        log::info!("engine_init: {} plates generated", sim.terrain.plates.len());
        let arrow_samples = arrow_sample_cells(&grid, ARROW_SAMPLES);
        let arrows = build_arrows_at(&grid, &sim.terrain.velocity, &arrow_samples);

        let mut renderer = Renderer::new(canvas0, canvas1, &mesh, grid.n)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        renderer.upload_plate_data(&sim.terrain.plate_id);
        renderer.upload_elevation(sim.elevations());
        renderer.upload_cell_centers(&build_cell_centers(&grid));
        renderer.set_arrows(&arrows);

        ENGINE.with(|cell| {
            *cell.borrow_mut() = Some(Engine {
                grid,
                sim,
                renderer,
                globe: OrbitCamera::default(),
                zoom: SurfaceCamera::default(),
                projection: [Projection::Sphere, Projection::Sphere],
                time_scale: DEFAULT_TIME_SCALE,
                arrow_samples,
                frame: 0,
            });
        });
        log::info!("engine_init: GPU ready");
        Ok(())
    }

    /// Regenerate the planet's terrain (plates + elevation) from a new `seed`, refreshing the
    /// static GPU data (plate ids, elevation, motion arrows). The climate state (temperatures) is
    /// left running, so the day/night cycle continues uninterrupted over the new geography.
    #[wasm_bindgen]
    pub fn engine_regenerate(seed: u32) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                // Spread the 32-bit UI seed across the 64-bit generator space.
                let seed = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ PLATE_SEED;
                engine.sim.generate_terrain(&engine.grid, NUM_PLATES, seed);
                engine.renderer.upload_plate_data(&engine.sim.terrain.plate_id);
                engine.renderer.upload_elevation(engine.sim.elevations());
                let arrows = build_arrows_at(
                    &engine.grid,
                    &engine.sim.terrain.velocity,
                    &engine.arrow_samples,
                );
                engine.renderer.set_arrows(&arrows);
            }
        });
    }

    /// Resize a view's surface (view index 0 = globe, 1 = zoomed).
    #[wasm_bindgen]
    pub fn engine_resize(view: usize, width: u32, height: u32) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                engine.renderer.resize(view, width, height);
            }
        });
    }

    /// Select the base data layer for a view (radio buttons; only "temperature" for now).
    #[wasm_bindgen]
    pub fn engine_set_layer(view: usize, layer: &str) {
        if let Some(layer) = Layer::from_str(layer) {
            ENGINE.with(|cell| {
                if let Some(engine) = cell.borrow_mut().as_mut() {
                    engine.renderer.set_layer(view, layer);
                }
            });
        }
    }

    /// Toggle an overlay ("sunlight" or "graticule") for a view from the UI.
    #[wasm_bindgen]
    pub fn engine_set_overlay(view: usize, which: &str, enabled: bool) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                engine.renderer.set_overlay(view, which, enabled);
            }
        });
    }

    /// Pan the zoomed view's target across the surface (east/north steps), from the arrow keys.
    #[wasm_bindgen]
    pub fn engine_pan_zoom(d_east: f32, d_north: f32) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                engine.zoom.pan(d_east, d_north);
            }
        });
    }

    /// Details about the cell under the cursor, returned to JS for the hover tooltip.
    #[wasm_bindgen]
    pub struct PickInfo {
        pub cell: u32,
        pub temp: f32,
        pub lon: f32,
        pub lat: f32,
        pub plate: u32,
        /// Surface elevation in metres relative to sea level (negative = below sea level).
        pub elev: f32,
    }

    /// Camera (view-projection + eye) for a view index: 0 = globe, 1 = zoomed. In a map
    /// projection the matrix is the flat-map orthographic transform and the eye is unused.
    fn view_camera(engine: &Engine, view: usize) -> Option<(glam::Mat4, glam::Vec3)> {
        let aspect = engine.renderer.aspect(view);
        let (base_proj, eye) = match view {
            0 => (engine.globe.view_proj(aspect), engine.globe.eye()),
            1 => (engine.zoom.view_proj(aspect), engine.zoom.eye()),
            _ => return None,
        };
        let proj = match engine.projection[view] {
            Projection::Sphere => base_proj,
            Projection::WinkelTripel => winkel_view_proj(aspect),
        };
        Some((proj, eye))
    }

    /// Turn an NDC cursor position into the surface point it points at, using whichever
    /// projection the view is currently drawn with.
    fn pick(engine: &Engine, view: usize, ndc: Vec2) -> Option<glam::Vec3> {
        let (view_proj, eye) = view_camera(engine, view)?;
        match engine.projection[view] {
            Projection::Sphere => ray_sphere(view_proj, eye, ndc),
            Projection::WinkelTripel => unproject_winkel(view_proj, ndc),
        }
    }

    /// Select the screen projection for a view ("sphere" or "winkel"), from the UI.
    #[wasm_bindgen]
    pub fn engine_set_projection(view: usize, kind: &str) {
        if view > 1 {
            return;
        }
        if let Some(proj) = Projection::from_str(kind) {
            ENGINE.with(|cell| {
                if let Some(engine) = cell.borrow_mut().as_mut() {
                    engine.projection[view] = proj;
                }
            });
        }
    }

    /// Pick the cell under the cursor in a view and highlight it. `ndc_x`/`ndc_y` are the
    /// cursor position in normalized device coordinates (each in [-1, 1], y pointing up).
    /// Returns details for the tooltip, or `None` if the ray misses the planet.
    #[wasm_bindgen]
    pub fn engine_hover(view: usize, ndc_x: f32, ndc_y: f32) -> Option<PickInfo> {
        ENGINE.with(|cell| {
            let mut guard = cell.borrow_mut();
            let engine = guard.as_mut()?;
            match pick(engine, view, Vec2::new(ndc_x, ndc_y)) {
                Some(hit) => {
                    let idx = engine.grid.nearest_cell(hit);
                    engine.renderer.set_highlight(view, Some(idx as u32));
                    let ll = engine.grid.lonlat_deg[idx];
                    let plate = engine.sim.terrain.plate_id.get(idx).copied().unwrap_or(0);
                    let elev = engine.sim.elevations().get(idx).copied().unwrap_or(0.0);
                    Some(PickInfo {
                        cell: idx as u32,
                        temp: engine.sim.temperatures()[idx],
                        lon: ll.x,
                        lat: ll.y,
                        plate,
                        elev,
                    })
                }
                None => {
                    engine.renderer.set_highlight(view, None);
                    None
                }
            }
        })
    }

    /// Clear a view's hover highlight (cursor left the canvas).
    #[wasm_bindgen]
    pub fn engine_clear_hover(view: usize) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                engine.renderer.set_highlight(view, None);
            }
        });
    }

    /// Recenter the zoomed view on the point clicked in the globe view (NDC coordinates).
    /// Ignores clicks that miss the planet.
    #[wasm_bindgen]
    pub fn engine_click_move(ndc_x: f32, ndc_y: f32) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                if let Some(hit) = pick(engine, 0, Vec2::new(ndc_x, ndc_y)) {
                    engine.zoom.target = hit;
                }
            }
        });
    }

    /// Set the fast-forward factor (simulation seconds advanced per real second). 0 fully pauses
    /// the simulation clock (rendering continues); 1 runs in real time. Driven by the UI speed
    /// slider.
    #[wasm_bindgen]
    pub fn engine_set_time_scale(scale: f32) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                engine.time_scale = scale.max(0.0);
            }
        });
    }

    /// Advance the simulation and render both views. Driven by the DOM thread's rAF tick;
    /// `dt_ms` is the wall-clock delta since the previous tick.
    #[wasm_bindgen]
    pub fn engine_tick(dt_ms: f64) {
        ENGINE.with(|cell| {
            let mut guard = cell.borrow_mut();
            let Some(engine) = guard.as_mut() else { return };

            engine.frame += 1;
            if engine.frame == 1 {
                log::info!("engine_tick: first frame rendering");
            }

            // Clamp the frame delta so a stall (e.g. tab backgrounded) can't jump the sim.
            let dt_real = (dt_ms as f32 / 1000.0).clamp(0.0, 0.1);

            // Fast-forward: advance the sim by real time scaled up by the current factor (0 when
            // paused). `advance` sub-steps internally to stay numerically stable.
            engine.sim.advance(&engine.grid, dt_real * engine.time_scale);
            engine.globe.azimuth += AUTO_ROTATE * dt_real;

            engine.renderer.upload_cell_data(engine.sim.temperatures());

            // Refresh the surface-wind arrow field from the live wind (diagnosed each step).
            let wind_arrows =
                build_arrows_at(&engine.grid, engine.sim.winds(), &engine.arrow_samples);
            engine.renderer.set_wind_arrows(&wind_arrows);

            let sun = engine.sim.sun_direction(engine.sim.time);

            // Outline of the zoomed view's footprint, drawn on the globe view.
            let marker = engine.zoom.footprint_outline(12);
            engine.renderer.set_marker(&marker);

            // Build each view's camera through `view_camera` so the active projection (3D globe
            // or flat map) is applied consistently with picking.
            let (globe_vp, globe_eye) = view_camera(engine, 0).unwrap();
            let (zoom_vp, zoom_eye) = view_camera(engine, 1).unwrap();
            let globe_cam = ViewCamera {
                view_proj: globe_vp,
                eye: globe_eye,
                projection: engine.projection[0],
            };
            let zoom_cam = ViewCamera {
                view_proj: zoom_vp,
                eye: zoom_eye,
                projection: engine.projection[1],
            };
            engine.renderer.render([globe_cam, zoom_cam], sun);
        });
    }
}
