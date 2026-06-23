// Engine worker: owns ALL GPU + simulation state. wgpu objects are !Send/!Sync once wasm
// atomics are enabled, so they must never leave this thread. wasm-bindgen-rayon spawns its
// own additional workers (for CPU compute) when initThreadPool runs.

import init, {
  initThreadPool,
  engine_init,
  engine_tick,
  engine_resize,
  engine_set_layer,
  engine_regenerate,
  engine_set_overlay,
  engine_set_projection,
  engine_pan_zoom,
  engine_set_time_scale,
  engine_hover,
  engine_clear_hover,
  engine_click_move,
} from '../pkg/ecology_simulator.js';

let initialized = false;

function post(type, extra = {}) {
  self.postMessage({ type, ...extra });
}

async function boot(canvases, hardwareConcurrency) {
  post('status', { text: 'loading wasm…' });
  await init(); // instantiate module + shared memory

  post('status', { text: `starting ${hardwareConcurrency} threads…` });
  await initThreadPool(hardwareConcurrency); // must come after init(), before any rayon work

  post('status', { text: 'initializing GPU…' });
  await engine_init(canvases[0], canvases[1]);

  initialized = true;
  post('ready', { text: `running · ${hardwareConcurrency} threads · WebGPU` });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init':
        await boot(msg.canvases, msg.hardwareConcurrency);
        break;
      case 'tick':
        if (initialized) {
          engine_tick(msg.dt);
          post('frame'); // ack → main thread schedules the next rAF
        }
        break;
      case 'resize':
        if (initialized) engine_resize(msg.view, msg.width, msg.height);
        break;
      case 'layer':
        if (initialized) engine_set_layer(msg.view, msg.layer);
        break;
      case 'regenerate':
        if (initialized) engine_regenerate(msg.seed >>> 0);
        break;
      case 'overlay':
        if (initialized) engine_set_overlay(msg.view, msg.which, msg.enabled);
        break;
      case 'projection':
        if (initialized) engine_set_projection(msg.view, msg.kind);
        break;
      case 'pan':
        if (initialized) engine_pan_zoom(msg.dEast, msg.dNorth);
        break;
      case 'speed':
        if (initialized) engine_set_time_scale(msg.scale);
        break;
      case 'hover':
        if (initialized) {
          const info = engine_hover(msg.view, msg.ndcX, msg.ndcY);
          // Marshal the wasm struct into a plain object, then free it.
          let plain = null;
          if (info) {
            plain = { cell: info.cell, temp: info.temp, lon: info.lon, lat: info.lat, plate: info.plate, elev: info.elev };
            info.free();
          }
          post('hoverInfo', { view: msg.view, info: plain });
        }
        break;
      case 'clearHover':
        if (initialized) engine_clear_hover(msg.view);
        break;
      case 'clickMove':
        if (initialized) engine_click_move(msg.ndcX, msg.ndcY);
        break;
    }
  } catch (err) {
    post('error', { text: String(err && err.message ? err.message : err) });
    // Re-throw so it also surfaces in the worker console with a stack.
    throw err;
  }
};
