// DOM main thread: owns the page + UI, transfers the two canvases to the engine worker as
// OffscreenCanvas, and drives the animation loop (workers have no requestAnimationFrame).
// All GPU + simulation work happens inside the worker (see web/worker.js).

const statusEl = document.getElementById('statusText');
const pauseBtn = document.getElementById('pause');
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
    pauseBtn.disabled = false;
    lastT = performance.now();
    requestAnimationFrame(loop);
  } else if (msg.type === 'frame') {
    // Worker finished a frame: schedule the next (natural backpressure).
    if (running) requestAnimationFrame(loop);
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

  worker.postMessage({ type: 'tick', dt: dtMs });
}

worker.postMessage(
  { type: 'init', canvases: offscreens, hardwareConcurrency: navigator.hardwareConcurrency || 4 },
  offscreens,
);

// Pause / resume the simulation clock (rendering + auto-rotation continue).
let paused = false;
pauseBtn.addEventListener('click', () => {
  paused = !paused;
  pauseBtn.textContent = paused ? '▶ Resume' : '⏸ Pause';
  worker.postMessage({ type: 'pause', paused });
});

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
