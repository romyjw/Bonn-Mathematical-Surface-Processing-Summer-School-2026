/* ===========================================================================
 *  classical_cpm.js
 *  ---------------------------------------------------------------------------
 *  Interactive 2-D Closest-Point Method demo on a random closed curve.
 *  Refactored from closest_point_method.html into a module attached to
 *  window.ClassicalCPM. Function names are preserved from the original.
 *
 *  In plain words, each iterate does three things:
 *    1) extend()      — copy u from the curve onto every band cell along
 *                       its closest-point assignment ("u constant along n̂")
 *    2) heatStep()    — explicit Euler with the 5-point grid Laplacian
 *    3) pullToCurve() — bilinear sample from the four nearest grid cells
 *                       back onto each curve point
 *  The new `setVisMode(mode)` toggles which of these stages is highlighted
 *  on the canvas so students can isolate each step visually.
 * ===========================================================================*/
(function () {

const cv  = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CW  = cv.width, CH = cv.height;

// ── tunables ──────────────────────────────────────────────────────────────
let h     = 12;
let bandS = 2;
let subN  = 8;
const DT  = 0.20;

// "mode": curve | cp | band | extended | heat | pulled
let visMode = "pulled";

// ── curve geometry ────────────────────────────────────────────────────────
const NC = 480;
const curvePts = [];
let curveNx, curveNy;
let curveU;
let curveParams;
let gridU_before_heat;   // snapshot for "heat diff" visualisation

// Random parametric closed curve: r(θ) = base + Σ amp·cos(freq θ + phase)
function randomCurveParams() {
  const nModes = 3 + Math.floor(Math.random() * 3);
  const modes = [];
  for (let i = 0; i < nModes; i++) {
    modes.push({
      freq:  2 + Math.floor(Math.random() * 6),
      amp:   (Math.random() - 0.35) * 50,
      phase: Math.random() * 2 * Math.PI,
    });
  }
  return {
    base:     150 + Math.random() * 40,
    modes,
    stretchX: 0.85 + Math.random() * 0.35,
    stretchY: 0.85 + Math.random() * 0.35,
    rot:      Math.random() * 2 * Math.PI,
  };
}

// Sample the parametric curve, rotate, fit into canvas with margin.
function buildCurve() {
  curvePts.length = 0;
  const p = curveParams;
  const raw = [];
  for (let k = 0; k < NC; k++) {
    const t = 2 * Math.PI * k / NC;
    let r = p.base;
    for (const m of p.modes) r += m.amp * Math.cos(m.freq * t + m.phase);
    r = Math.max(50, r);
    raw.push({ x: r * Math.cos(t) * p.stretchX, y: r * Math.sin(t) * p.stretchY });
  }
  const cr = Math.cos(p.rot), sr = Math.sin(p.rot);
  for (const pt of raw) {
    const nx = pt.x * cr - pt.y * sr;
    const ny = pt.x * sr + pt.y * cr;
    pt.x = nx; pt.y = ny;
  }
  let mnX=Infinity, mxX=-Infinity, mnY=Infinity, mxY=-Infinity;
  for (const pt of raw) {
    if (pt.x < mnX) mnX = pt.x; if (pt.x > mxX) mxX = pt.x;
    if (pt.y < mnY) mnY = pt.y; if (pt.y > mxY) mxY = pt.y;
  }
  const wantW = CW * 0.74, wantH = CH * 0.74;
  const sc = Math.min(wantW / (mxX - mnX), wantH / (mxY - mnY));
  const ccx = CW / 2 - 0.5 * (mnX + mxX) * sc;
  const ccy = CH / 2 - 0.5 * (mnY + mxY) * sc;
  for (const pt of raw) curvePts.push({ x: ccx + pt.x * sc, y: ccy + pt.y * sc });
}

// Outward unit normal at each curve point (central-difference tangent, then rotate -90°).
function computeNormals() {
  curveNx = new Float32Array(NC);
  curveNy = new Float32Array(NC);
  for (let k = 0; k < NC; k++) {
    const kp = (k + 1) % NC, km = (k - 1 + NC) % NC;
    const tx = curvePts[kp].x - curvePts[km].x;
    const ty = curvePts[kp].y - curvePts[km].y;
    const L = Math.hypot(tx, ty) || 1;
    curveNx[k] =  ty / L;
    curveNy[k] = -tx / L;
  }
  // flip to outward
  let cx = 0, cy = 0;
  for (const p of curvePts) { cx += p.x; cy += p.y; }
  cx /= NC; cy /= NC;
  const dot = (curvePts[0].x - cx) * curveNx[0] + (curvePts[0].y - cy) * curveNy[0];
  if (dot < 0) {
    for (let k = 0; k < NC; k++) { curveNx[k] = -curveNx[k]; curveNy[k] = -curveNy[k]; }
  }
}

// Random superposition of low-freq sines on the curve, then normalise to ±1.
function initU() {
  curveU = new Float32Array(NC);
  const M = 5;
  const amp   = Array.from({ length: M }, () => (Math.random() - 0.5) * 2.0);
  const phase = Array.from({ length: M }, () => Math.random() * 2 * Math.PI);
  let mx = 0;
  for (let k = 0; k < NC; k++) {
    const t = 2 * Math.PI * k / NC;
    let v = 0;
    for (let m = 0; m < M; m++) v += amp[m] * Math.sin((m + 1) * t + phase[m]);
    curveU[k] = v;
    mx = Math.max(mx, Math.abs(v));
  }
  if (mx > 0) for (let k = 0; k < NC; k++) curveU[k] /= mx;
}

// ── grid bookkeeping ─────────────────────────────────────────────────────
let NI, NJ;
let cpIdx;     // curve index assigned to each grid node (-1 outside band)
let cpDist2;   // squared distance to that curve point
let gridU;
let workU;
function gid(i, j) { return i * NJ + j; }

// Construct the narrow band:
//  (a) for every curve sample march ±n̂ within distance s·h, painting grid
//      nodes with that curve's index (closest wins on conflict);
//  (b) closest-point fallback for any remaining gap.
function buildGrid() {
  NI = Math.ceil(CW / h) + 2;
  NJ = Math.ceil(CH / h) + 2;
  cpIdx   = new Int32Array(NI * NJ).fill(-1);
  cpDist2 = new Float32Array(NI * NJ).fill(Infinity);
  gridU   = new Float32Array(NI * NJ);
  workU   = new Float32Array(NI * NJ);

  const maxD  = bandS * h;
  const maxD2 = maxD * maxD;
  const step  = Math.min(h * 0.4, 1.5);
  const nMarch = Math.ceil(maxD / step);

  for (let k = 0; k < NC; k++) {
    const px = curvePts[k].x, py = curvePts[k].y;
    const nx = curveNx[k],    ny = curveNy[k];
    for (let dir = -1; dir <= 1; dir += 2) {
      for (let si = 0; si <= nMarch; si++) {
        const d = si * step;
        const x = px + dir * d * nx;
        const y = py + dir * d * ny;
        const i = Math.round(x / h);
        const j = Math.round(y / h);
        if (i < 0 || i >= NI || j < 0 || j >= NJ) continue;
        const id = gid(i, j);
        const dx = i * h - px, dy = j * h - py;
        const d2 = dx * dx + dy * dy;
        if (d2 <= maxD2 && d2 < cpDist2[id]) {
          cpDist2[id] = d2;
          cpIdx[id]   = k;
        }
      }
    }
  }
  for (let i = 0; i < NI; i++) {
    for (let j = 0; j < NJ; j++) {
      const id = gid(i, j);
      if (cpIdx[id] >= 0) continue;
      const x = i * h, y = j * h;
      let best = maxD2, bk = -1;
      for (let k = 0; k < NC; k++) {
        const dx = x - curvePts[k].x, dy = y - curvePts[k].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; bk = k; }
      }
      if (bk >= 0) { cpIdx[id] = bk; cpDist2[id] = best; }
    }
  }
}

// ── PDE primitives ───────────────────────────────────────────────────────

// Step ①: u(band) ← u(curve) at the assigned closest curve point.
function extend() {
  for (let n = 0; n < NI * NJ; n++) {
    const k = cpIdx[n];
    gridU[n] = k >= 0 ? curveU[k] : 0;
  }
}

// Step ②: one explicit-Euler heat update with the 5-point stencil. Missing
// neighbours fall back to self so the boundary does not poison the interior.
function heatStep() {
  for (let i = 0; i < NI; i++) {
    for (let j = 0; j < NJ; j++) {
      const id = gid(i, j);
      if (cpIdx[id] < 0) { workU[id] = 0; continue; }
      const u = gridU[id];
      const uE = (i + 1 < NI && cpIdx[gid(i+1,j)] >= 0) ? gridU[gid(i+1,j)] : u;
      const uW = (i - 1 >= 0 && cpIdx[gid(i-1,j)] >= 0) ? gridU[gid(i-1,j)] : u;
      const uN = (j + 1 < NJ && cpIdx[gid(i,j+1)] >= 0) ? gridU[gid(i,j+1)] : u;
      const uS = (j - 1 >= 0 && cpIdx[gid(i,j-1)] >= 0) ? gridU[gid(i,j-1)] : u;
      workU[id] = u + DT * (uE + uW + uN + uS - 4 * u);
    }
  }
  for (let n = 0; n < NI * NJ; n++) if (cpIdx[n] >= 0) gridU[n] = workU[n];
}

// Step ③: bilinear sample of the grid back onto each curve point (fallback
// to nearest-neighbour if any of the four corners is outside the band).
function pullToCurve() {
  for (let k = 0; k < NC; k++) {
    const fx = curvePts[k].x / h;
    const fy = curvePts[k].y / h;
    const i0 = Math.floor(fx), j0 = Math.floor(fy);
    const i1 = i0 + 1,          j1 = j0 + 1;
    const tx = fx - i0,          ty = fy - j0;
    if (i0 < 0 || i1 >= NI || j0 < 0 || j1 >= NJ) continue;
    const a00 = cpIdx[gid(i0,j0)] >= 0;
    const a10 = cpIdx[gid(i1,j0)] >= 0;
    const a01 = cpIdx[gid(i0,j1)] >= 0;
    const a11 = cpIdx[gid(i1,j1)] >= 0;
    if (a00 && a10 && a01 && a11) {
      curveU[k] =
        (1-tx)*(1-ty)*gridU[gid(i0,j0)] +
            tx*(1-ty)*gridU[gid(i1,j0)] +
        (1-tx)*    ty*gridU[gid(i0,j1)] +
            tx*    ty*gridU[gid(i1,j1)];
    } else {
      const ni = Math.round(fx), nj = Math.round(fy);
      if (ni >= 0 && ni < NI && nj >= 0 && nj < NJ && cpIdx[gid(ni,nj)] >= 0)
        curveU[k] = gridU[gid(ni,nj)];
    }
  }
}

// ── colormap ─────────────────────────────────────────────────────────────
// Magma colormap (matplotlib), sampled at 9 evenly-spaced stops. We linearly
// interpolate between stops for an inexpensive approximation that still
// looks identical to the eye.
const MAGMA = [
  [0.001, 0.000, 0.014],
  [0.080, 0.062, 0.230],
  [0.230, 0.060, 0.437],
  [0.385, 0.121, 0.508],
  [0.557, 0.221, 0.520],
  [0.747, 0.292, 0.483],
  [0.945, 0.470, 0.435],
  [0.988, 0.712, 0.430],
  [0.987, 0.991, 0.750],
];
function magma(t) {
  t = Math.max(0, Math.min(1, t));
  const x  = t * (MAGMA.length - 1);
  const i  = Math.floor(x);
  const f  = x - i;
  const a  = MAGMA[i];
  const b  = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
  return [a[0] + (b[0]-a[0]) * f,
          a[1] + (b[1]-a[1]) * f,
          a[2] + (b[2]-a[2]) * f];
}
// Map a value in [-1, 1] to a CSS rgb() string via the magma colormap.
function rgb(v) {
  v = Math.max(-1, Math.min(1, v));
  const [r, g, b] = magma((v + 1) * 0.5);
  return `rgb(${(r*255)|0},${(g*255)|0},${(b*255)|0})`;
}

// ── rendering ────────────────────────────────────────────────────────────
let showNormals = false;

function drawCurveOnly() {
  ctx.lineWidth = 5; ctx.lineCap = 'round';
  for (let k = 0; k < NC; k++) {
    const k2 = (k + 1) % NC;
    ctx.strokeStyle = rgb((curveU[k] + curveU[k2]) * 0.5);
    ctx.beginPath();
    ctx.moveTo(curvePts[k].x, curvePts[k].y);
    ctx.lineTo(curvePts[k2].x, curvePts[k2].y);
    ctx.stroke();
  }
}

// Visualise the closest-point map by drawing short red segments from a
// subset of band cells to their assigned curve point.
function drawCPArrows() {
  const stride = Math.max(2, Math.floor(NI*NJ / 600));
  ctx.strokeStyle = 'rgba(192, 57, 43, 0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < NI; i++) {
    for (let j = 0; j < NJ; j++) {
      const id = gid(i, j);
      if (cpIdx[id] < 0) continue;
      if ((i + j) % stride !== 0) continue;
      const k = cpIdx[id];
      ctx.moveTo(i * h, j * h);
      ctx.lineTo(curvePts[k].x, curvePts[k].y);
    }
  }
  ctx.stroke();
}

// Outline-only band cells (no colour fill) — to show shape without u.
function drawBandOutline() {
  const cell = Math.max(1, h - 1);
  ctx.fillStyle = 'rgba(126, 200, 227, 0.18)';
  for (let i = 0; i < NI; i++) {
    const x = i * h;
    for (let j = 0; j < NJ; j++) {
      if (cpIdx[gid(i,j)] < 0) continue;
      ctx.fillRect(x - cell * 0.5, j * h - cell * 0.5, cell, cell);
    }
  }
}

// Colour band cells by the *current* gridU (used for "extended" and "pulled").
function drawBandFilled() {
  const cell = Math.max(1, h - 1);
  ctx.globalAlpha = 0.6;
  for (let i = 0; i < NI; i++) {
    const x = i * h;
    for (let j = 0; j < NJ; j++) {
      if (cpIdx[gid(i,j)] < 0) continue;
      ctx.fillStyle = rgb(gridU[gid(i,j)]);
      ctx.fillRect(x - cell * 0.5, j * h - cell * 0.5, cell, cell);
    }
  }
  ctx.globalAlpha = 1;
}

// Colour cells by (gridU - gridU_before_heat), the change made by one
// heatStep. Useful to *see* the diffusion operator at work.
function drawHeatDiff() {
  if (!gridU_before_heat) return;
  const cell = Math.max(1, h - 1);
  ctx.globalAlpha = 0.8;
  for (let i = 0; i < NI; i++) {
    const x = i * h;
    for (let j = 0; j < NJ; j++) {
      const id = gid(i, j);
      if (cpIdx[id] < 0) continue;
      const d = (gridU[id] - gridU_before_heat[id]) * 4;
      ctx.fillStyle = rgb(d);
      ctx.fillRect(x - cell * 0.5, j * h - cell * 0.5, cell, cell);
    }
  }
  ctx.globalAlpha = 1;
}

function drawNormals() {
  const stride = Math.max(1, Math.floor(NC / 80));
  const Ln = bandS * h;
  ctx.strokeStyle = 'rgba(255,255,255,0.20)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k < NC; k += stride) {
    const px = curvePts[k].x, py = curvePts[k].y;
    const nx = curveNx[k],    ny = curveNy[k];
    ctx.moveTo(px - Ln * nx, py - Ln * ny);
    ctx.lineTo(px + Ln * nx, py + Ln * ny);
  }
  ctx.stroke();
}

// Master draw: composes the right layers for the current visMode.
function draw() {
  ctx.clearRect(0, 0, CW, CH);
  ctx.fillStyle = '#0d0d1f';
  ctx.fillRect(0, 0, CW, CH);

  switch (visMode) {
    case "curve":
      drawCurveOnly();
      break;
    case "cp":
      drawBandOutline();
      drawCPArrows();
      drawCurveOnly();
      break;
    case "band":
      drawBandOutline();
      drawCurveOnly();
      break;
    case "extended":
      drawBandFilled();        // gridU right after extend()
      drawCurveOnly();
      break;
    case "heat":
      drawHeatDiff();
      drawCurveOnly();
      break;
    case "pulled":
    default:
      drawBandFilled();
      drawCurveOnly();
      break;
  }
  if (showNormals) drawNormals();

  ctx.fillStyle = '#334';
  ctx.font = '11px Courier New';
  ctx.fillText(
    `h=${h}px  band s=${bandS.toFixed(1)} (±${(bandS*h).toFixed(0)}px)  ` +
    `grid ${NI}×${NJ}  DT=${DT}  sub-steps=${subN}  step ${stepN}  mode=${visMode}`,
    8, CH - 8
  );
}

// ── main loop ────────────────────────────────────────────────────────────
let stepN  = 0;
let autoOn = false;
let autoTick = null;

function doIterate() {
  extend();
  // Snapshot for "heat diff" visualisation BEFORE applying the heat sub-steps.
  if (!gridU_before_heat || gridU_before_heat.length !== gridU.length) {
    gridU_before_heat = new Float32Array(gridU.length);
  }
  gridU_before_heat.set(gridU);

  for (let s = 0; s < subN; s++) heatStep();
  pullToCurve();
  stepN++;
  document.getElementById('status').textContent = `step ${stepN}`;
  highlightAlgoStep(visMode);
  draw();
}

function freshU() {
  if (autoOn) toggleAuto();
  stepN = 0;
  document.getElementById('status').textContent = 'step 0';
  initU();
  extend();
  draw();
}

function newCurve() {
  if (autoOn) toggleAuto();
  stepN = 0;
  document.getElementById('status').textContent = 'step 0';
  curveParams = randomCurveParams();
  buildCurve();
  computeNormals();
  buildGrid();
  initU();
  extend();
  draw();
}

function toggleAuto() {
  autoOn = !autoOn;
  const btn = document.getElementById('btnAuto');
  btn.classList.toggle('on', autoOn);
  btn.textContent = autoOn ? 'Stop ■' : 'Auto ▶';
  if (autoOn) autoTick = setInterval(doIterate, 80);
  else        clearInterval(autoTick);
}

function bindSlider(id, valId, parse, fn) {
  document.getElementById(id).addEventListener('input', function () {
    const v = parse(this.value);
    document.getElementById(valId).textContent = Number.isInteger(v) ? v : v.toFixed(1);
    fn(v);
  });
}

// Public: switch which stage of the algorithm is highlighted.
function setVisMode(mode) {
  visMode = mode;
  highlightAlgoStep(mode);
  draw();
}

// Highlight the matching ①/②/③ token in the description below the canvas.
function highlightAlgoStep(mode) {
  const map = { cp: "step-1", band: "step-1", extended: "step-1",
                heat: "step-2", pulled: "step-3", curve: null };
  ["step-1", "step-2", "step-3"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', map[mode] === id);
  });
}

bindSlider('slBand', 'vBand', parseFloat, v => { bandS = v; buildGrid(); extend(); draw(); });
bindSlider('slH',    'vH',    parseInt,   v => { h = v;     buildGrid(); extend(); draw(); });
bindSlider('slSub',  'vSub',  parseInt,   v => { subN = v; });

document.getElementById('btnNew').addEventListener('click', newCurve);
document.getElementById('btnReset').addEventListener('click', freshU);
document.getElementById('btnIter').addEventListener('click', doIterate);
document.getElementById('btnAuto').addEventListener('click', toggleAuto);
document.getElementById('cbNormals').addEventListener('change', function () {
  showNormals = this.checked; draw();
});
document.querySelectorAll('input[name="vismode"]').forEach(el => {
  el.addEventListener('change', () => { if (el.checked) setVisMode(el.value); });
});

// ── init ─────────────────────────────────────────────────────────────────
curveParams = randomCurveParams();
buildCurve();
computeNormals();
buildGrid();
initU();
extend();
highlightAlgoStep(visMode);
draw();

window.ClassicalCPM = { setVisMode, doIterate, newCurve, freshU };

})();
