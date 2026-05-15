/* ===========================================================================
 *  neural_cpm.js
 *  ---------------------------------------------------------------------------
 *  Two Three.js point-cloud viewers (Poisson stepper + Heat animation) for
 *  the Apple example. All heavy lifting (CPM band, neural rot_update, RBF,
 *  sparse solve, time-stepping) was done offline by `export_for_web.py`
 *  and shipped as JSON under ../precomputed/.
 *
 *  Each viewer just colours a fixed set of points by a 1-D scalar field
 *  using the same blue-white-red colormap as the 2-D demo, so students
 *  see "the same maths, on a real 3-D surface".
 * ===========================================================================*/
(function () {

const PRE = "../precomputed/";

// Magma colormap (matplotlib), sampled at 9 stops with linear interp.
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
function rgbFor(v, vmin, vmax) {
  if (vmax === vmin) { return [1, 1, 1]; }
  const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  const x = t * (MAGMA.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = MAGMA[i];
  const b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
  return [a[0] + (b[0]-a[0]) * f,
          a[1] + (b[1]-a[1]) * f,
          a[2] + (b[2]-a[2]) * f];
}

// Common scene factory: returns {scene, camera, renderer, controls, animate}.
function makeViewer(containerId, radius) {
  const container = document.getElementById(containerId);
  const w = container.clientWidth, h = container.clientHeight;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d1f);

  const camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 100);
  camera.position.set(radius * 1.8, radius * 1.2, radius * 1.8);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);

  function onResize() {
    const W = container.clientWidth, H = container.clientHeight;
    camera.aspect = W / H; camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }
  window.addEventListener('resize', onResize);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return { scene, camera, renderer, controls };
}

// Build a THREE.Points cloud from a list of [x,y,z] coordinates.
function makePointCloud(positions, ptSize) {
  const N = positions.length;
  const posArr = new Float32Array(N * 3);
  const colArr = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    posArr[3*i  ] = positions[i][0];
    posArr[3*i+1] = positions[i][1];
    posArr[3*i+2] = positions[i][2];
    colArr[3*i  ] = 0.9; colArr[3*i+1] = 0.9; colArr[3*i+2] = 0.9;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  geom.setAttribute('color',    new THREE.BufferAttribute(colArr, 3));
  const mat  = new THREE.PointsMaterial({
    size: ptSize, vertexColors: true, sizeAttenuation: true,
  });
  return new THREE.Points(geom, mat);
}

// Repaint vertex colors of a Points object from a 1-D scalar array.
function colorByValues(points, values, vmin, vmax) {
  const col = points.geometry.attributes.color.array;
  for (let i = 0; i < values.length; i++) {
    const [r, g, b] = rgbFor(values[i], vmin, vmax);
    col[3*i] = r; col[3*i+1] = g; col[3*i+2] = b;
  }
  points.geometry.attributes.color.needsUpdate = true;
}

// ── load & wire up everything ────────────────────────────────────────────
async function main() {
  let meta, mesh, poisson, heat;
  try {
    [meta, mesh, poisson, heat] = await Promise.all([
      fetch(PRE + "meta.json").then(r => r.json()),
      fetch(PRE + "apple_mesh.json").then(r => r.json()),
      fetch(PRE + "poisson_frames.json").then(r => r.json()),
      fetch(PRE + "heat_frames.json").then(r => r.json()),
    ]);
  } catch (e) {
    document.getElementById('nstatus').textContent =
      "Failed to load precomputed JSON. Serve from neural-cpm/ via `python -m http.server` "
      + "and open  /web/index.html . Error: " + e.message;
    return;
  }

  // ---- Poisson viewer ----
  const pView   = makeViewer('three-poisson', meta.radius);
  const surfPts = makePointCloud(mesh.surface_points, 0.015);
  const bandPts = makePointCloud(mesh.band_points,    0.020);
  pView.scene.add(surfPts);
  pView.scene.add(bandPts);

  let pStage = 0;
  const pCap  = document.getElementById('pCaption');
  const pInd  = document.getElementById('pStageIndicator');

  function showPoissonStage(idx) {
    pStage = (idx + poisson.length) % poisson.length;
    const s = poisson[pStage];
    if (s.target === 'band') {
      bandPts.visible = true;
      surfPts.visible = true;
      // dim the surface in band stages
      const c = surfPts.geometry.attributes.color.array;
      for (let i = 0; i < c.length; i++) c[i] = 0.25;
      surfPts.geometry.attributes.color.needsUpdate = true;
      colorByValues(bandPts, s.values, s.vmin, s.vmax);
    } else {
      bandPts.visible = false;
      surfPts.visible = true;
      colorByValues(surfPts, s.values, s.vmin, s.vmax);
    }
    pCap.innerHTML = `<strong>Stage ${pStage + 1} / ${poisson.length}:</strong> `
                   + s.name + " &mdash; " + (s.caption || "");
    pInd.textContent = `${pStage + 1}/${poisson.length}`;
  }

  document.getElementById('pPrev').onclick = () => showPoissonStage(pStage - 1);
  document.getElementById('pNext').onclick = () => showPoissonStage(pStage + 1);
  showPoissonStage(0);

  // ---- Heat viewer ----
  const hView   = makeViewer('three-heat', meta.radius);
  const hSurfPts = makePointCloud(mesh.surface_points, 0.015);
  hView.scene.add(hSurfPts);

  const slider   = document.getElementById('hSlider');
  const hCap     = document.getElementById('hCaption');
  const hStepLbl = document.getElementById('hStepLabel');
  const btnPlay  = document.getElementById('hPlay');
  slider.min = 0;
  slider.max = heat.n_frames - 1;
  slider.value = 0;

  function showHeatFrame(idx) {
    idx = Math.max(0, Math.min(heat.n_frames - 1, idx | 0));
    slider.value = idx;
    colorByValues(hSurfPts, heat.values[idx], heat.vmin, heat.vmax);
    const t  = heat.step_times ? heat.step_times[idx] : idx;
    hStepLbl.textContent = `frame ${idx}/${heat.n_frames - 1}  (step ${t})`;
    hCap.innerHTML =
      `<strong>Heat diffusion on the Apple surface.</strong> `
      + `Each frame is one explicit-Euler step of <em>Δu = ∂u/∂t</em> with the `
      + `surface Laplacian; the neural rot_update keeps u faithful to the surface.`;
  }

  slider.addEventListener('input', () => showHeatFrame(parseInt(slider.value, 10)));

  let playing = false, playTimer = null;
  btnPlay.onclick = () => {
    playing = !playing;
    btnPlay.classList.toggle('on', playing);
    btnPlay.textContent = playing ? 'Pause ■' : 'Play ▶';
    if (playing) {
      playTimer = setInterval(() => {
        let v = parseInt(slider.value, 10) + 1;
        if (v > heat.n_frames - 1) v = 0;
        showHeatFrame(v);
      }, 120);
    } else clearInterval(playTimer);
  };

  showHeatFrame(0);
  document.getElementById('nstatus').textContent =
    `loaded ${mesh.surface_points.length} surface pts, `
    + `${mesh.band_points.length} band pts, `
    + `${heat.n_frames} heat frames.`;
}

// Wait for DOM
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else { main(); }

})();
