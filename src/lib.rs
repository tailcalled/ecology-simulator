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
pub mod sim;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;
    use web_sys::OffscreenCanvas;

    use crate::grid::Grid;
    use crate::render::camera::{OrbitCamera, SurfaceCamera};
    use crate::render::mesh::build_mesh;
    use crate::render::{Layer, Renderer, ViewCamera};
    use crate::sim::{Climate, Sim};

    // Re-export the rayon thread-pool initializer; wasm-bindgen exposes it to JS as
    // `initThreadPool`. JS must call it (after `init()`) before any rayon work runs.
    #[allow(unused_imports)]
    pub use wasm_bindgen_rayon::init_thread_pool;

    /// Number of grid cells. ~16k gives good resolution while keeping the one-time Voronoi
    /// precompute and the mesh size modest.
    const GRID_CELLS: usize = 16_384;
    /// Simulation seconds advanced per real second — a day (86 400 s) passes in ~14 s.
    const TIME_SCALE: f32 = 6_000.0;
    /// Big-picture view auto-rotation (rad/s).
    const AUTO_ROTATE: f32 = 0.08;
    /// Starting temperature for every cell (K).
    const INITIAL_TEMP: f32 = 255.0;

    /// The whole engine: grid + simulation + renderer + the two cameras. Lives in a worker
    /// thread-local because the renderer's GPU objects cannot cross threads.
    struct Engine {
        grid: Grid,
        sim: Sim,
        renderer: Renderer,
        globe: OrbitCamera,
        zoom: SurfaceCamera,
        paused: bool,
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

        let sim = Sim::new(grid.n, Climate::default(), INITIAL_TEMP);
        let renderer = Renderer::new(canvas0, canvas1, &mesh, grid.n)
            .await
            .map_err(|e| JsValue::from_str(&e))?;

        ENGINE.with(|cell| {
            *cell.borrow_mut() = Some(Engine {
                grid,
                sim,
                renderer,
                globe: OrbitCamera::default(),
                zoom: SurfaceCamera::default(),
                paused: false,
                frame: 0,
            });
        });
        log::info!("engine_init: GPU ready");
        Ok(())
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

    /// Pause/resume the simulation clock (rendering continues).
    #[wasm_bindgen]
    pub fn engine_set_paused(paused: bool) {
        ENGINE.with(|cell| {
            if let Some(engine) = cell.borrow_mut().as_mut() {
                engine.paused = paused;
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

            if !engine.paused {
                engine.sim.advance(&engine.grid, dt_real * TIME_SCALE);
            }
            engine.globe.azimuth += AUTO_ROTATE * dt_real;

            engine.renderer.upload_cell_data(engine.sim.temperatures());

            let sun = engine.sim.sun_direction(engine.sim.time);

            // Outline of the zoomed view's footprint, drawn on the globe view.
            let marker = engine.zoom.footprint_outline(12);
            engine.renderer.set_marker(&marker);

            let globe_cam = ViewCamera {
                view_proj: engine.globe.view_proj(engine.renderer.aspect(0)),
                eye: engine.globe.eye(),
            };
            let zoom_cam = ViewCamera {
                view_proj: engine.zoom.view_proj(engine.renderer.aspect(1)),
                eye: engine.zoom.eye(),
            };
            engine.renderer.render([globe_cam, zoom_cam], sun);
        });
    }
}
