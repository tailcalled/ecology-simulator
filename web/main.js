// DOM main thread: owns the page + UI, transfers the two canvases to the engine worker as
// OffscreenCanvas, and drives the animation loop (workers have no requestAnimationFrame).
// All GPU + simulation work happens inside the worker (see web/worker.js).

const statusEl = document.getElementById('statusText');
const speedSlider = document.getElementById('speedSlider');
const speedLabel = document.getElementById('speedLabel');
const tooltipEl = document.getElementById('tooltip');
const setStatus = (msg) => { statusEl.textContent = msg; };

if (!crossOriginIsolated) {
  setStatus('NOT cross-origin isolated — SharedArrayBuffer/threads unavailable. ' +
            'Serve via serve.py (sets COOP/COEP).');
  throw new Error('not crossOriginIsolated');
}
if (!('gpu' in navigator)) {
  setStatus('WebGPU not available in this browser.');
  throw new Error('no WebGPU');
}

const dpr = Math.min(self.devicePixelRatio || 1, 2);
const canvasEls = [document.getElementById('globe'), document.getElementById('zoom')];

// Size the drawing buffers before transfer; after transferControlToOffscreen the element's
// width/height become read-only on the main thread (resizes go through the worker).
function backingSize(el) {
  const r = el.getBoundingClientRect();
  return [Math.max(1, Math.round(r.width * dpr)), Math.max(1, Math.round(r.height * dpr))];
}

const offscreens = canvasEls.map((el) => {
  const [w, h] = backingSize(el);
  el.width = w; el.height = h;
  return el.transferControlToOffscreen();
});

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

let running = false;
let lastT = 0;

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'status') {
    setStatus(msg.text);
  } else if (msg.type === 'ready') {
    setStatus(msg.text);
    running = true;
    speedSlider.disabled = false;
    lastT = performance.now();
    requestAnimationFrame(loop);
  } else if (msg.type === 'frame') {
    // Worker finished a frame: schedule the next (natural backpressure).
    if (running) requestAnimationFrame(loop);
  } else if (msg.type === 'hoverInfo') {
    updateTooltip(msg.info);
  } else if (msg.type === 'error') {
    setStatus('error: ' + msg.text);
    running = false;
  }
};

// Continuous panning: apply currently-held arrow keys every frame (frame-rate independent),
// rather than relying on discrete key-repeat events.
const held = new Set();
const PAN_SPEED = 0.2; // radians/second across the surface

function loop(t) {
  const dtMs = t - lastT;
  lastT = t;

  let dE = 0;
  let dN = 0;
  if (held.has('ArrowLeft')) dE -= 1;
  if (held.has('ArrowRight')) dE += 1;
  if (held.has('ArrowUp')) dN += 1;
  if (held.has('ArrowDown')) dN -= 1;
  if (dE || dN) {
    const s = PAN_SPEED * (dtMs / 1000);
    worker.postMessage({ type: 'pan', dEast: dE * s, dNorth: dN * s });
  }

  // Coalesce cursor moves to at most one pick request per frame.
  if (pendingHover) {
    worker.postMessage(pendingHover);
    pendingHover = null;
  }

  worker.postMessage({ type: 'tick', dt: dtMs });
}

worker.postMessage(
  { type: 'init', canvases: offscreens, hardwareConcurrency: navigator.hardwareConcurrency || 4 },
  offscreens,
);

// Simulation speed: an almost-logarithmic ladder of fast-forward factors (sim seconds per real
// second). The two lowest rungs are special-cased — 0 fully pauses, 1 is real time — and the
// rest climb a 1-2-5 decade ladder. The slider's min/max in the HTML must match this length.
const SPEEDS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];

function speedText(s) {
  if (s === 0) return 'Paused';
  if (s === 1) return 'Real-time';
  return `${s.toLocaleString()}×`;
}

function applySpeed() {
  const scale = SPEEDS[Number(speedSlider.value)];
  speedLabel.textContent = speedText(scale);
  worker.postMessage({ type: 'speed', scale });
}
speedSlider.addEventListener('input', applySpeed);

// Forward base-layer selection (radio buttons) per view.
for (const r of document.querySelectorAll('input.layer')) {
  r.addEventListener('change', () => {
    if (r.checked) {
      worker.postMessage({ type: 'layer', view: Number(r.dataset.view), layer: r.value });
    }
  });
}

// Forward overlay toggles (Sunlight / Lat-long lines) per view to the worker.
for (const cb of document.querySelectorAll('input.ov')) {
  cb.addEventListener('change', () => {
    worker.postMessage({
      type: 'overlay',
      view: Number(cb.dataset.view),
      which: cb.dataset.ov,
      enabled: cb.checked,
    });
  });
}

// --- Cursor hover (highlight + details box) and click-to-recenter ---
// The latest cursor pick, sent to the worker once per frame in loop(); the worker replies with
// 'hoverInfo' carrying the cell details (or null if the ray missed the planet).
let pendingHover = null;

// Convert a pointer event over `el` to normalized device coords (x, y in [-1, 1], y up).
function eventToNdc(el, e) {
  const r = el.getBoundingClientRect();
  return [
    ((e.clientX - r.left) / r.width) * 2 - 1,
    1 - ((e.clientY - r.top) / r.height) * 2,
  ];
}

function updateTooltip(info) {
  if (!info) {
    tooltipEl.style.display = 'none';
    return;
  }
  const c = (info.temp - 273.15).toFixed(1);
  tooltipEl.innerHTML =
    `<div class="temp">${info.temp.toFixed(1)} K · ${c} °C</div>` +
    `<div class="coord">Plate ${info.plate} · ${info.lat.toFixed(1)}°, ${info.lon.toFixed(1)}°</div>`;
  tooltipEl.style.display = 'block';
}

canvasEls.forEach((el, view) => {
  el.addEventListener('mousemove', (e) => {
    const [ndcX, ndcY] = eventToNdc(el, e);
    pendingHover = { type: 'hover', view, ndcX, ndcY };
    // Position the box at the cursor immediately so it tracks smoothly.
    tooltipEl.style.left = `${e.clientX + 14}px`;
    tooltipEl.style.top = `${e.clientY + 14}px`;
  });
  el.addEventListener('mouseleave', () => {
    pendingHover = null;
    tooltipEl.style.display = 'none';
    worker.postMessage({ type: 'clearHover', view });
  });
});

// Clicking the planet view recenters the zoomed view on that spot.
canvasEls[0].addEventListener('click', (e) => {
  const [ndcX, ndcY] = eventToNdc(canvasEls[0], e);
  worker.postMessage({ type: 'clickMove', ndcX, ndcY });
});

// Track held arrow keys; the actual panning is applied continuously in loop().
const PAN_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']);
window.addEventListener('keydown', (e) => {
  if (!PAN_KEYS.has(e.key)) return;
  e.preventDefault(); // don't scroll the page
  held.add(e.key);
});
window.addEventListener('keyup', (e) => held.delete(e.key));
window.addEventListener('blur', () => held.clear()); // stop if focus leaves the page

// Forward resizes; the worker owns the OffscreenCanvas backing-store size.
const ro = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const idx = canvasEls.indexOf(entry.target);
    if (idx < 0) continue;
    const [w, h] = backingSize(entry.target);
    worker.postMessage({ type: 'resize', view: idx, width: w, height: h });
  }
});
canvasEls.forEach((el) => ro.observe(el));
