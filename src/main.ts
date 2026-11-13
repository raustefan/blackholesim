import traceCode from "./trace.wgsl?raw";
import postCode from "./post.wgsl?raw";
import { DISK_N, FLAGS, Renderer, type TraceParams } from "./renderer.ts";
import { loadDiffuse, loadStars, skyRotation } from "./sky.ts";
import { Observer, position, zamoOmega, type Mode } from "./observer.ts";
import { SI, diskTemperature, eddingtonLuminosity, ergosphere, horizonInner, horizonOuter, isco, photonOrbit, rg, tg } from "./kerr.ts";
import { drawChart } from "./chart.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("view");
const overlay = $("overlay");

function showOverlay(title: string, body: string, button?: string) {
  $("overlay-title").textContent = title;
  $("overlay-body").innerHTML = body;
  const b = $<HTMLButtonElement>("overlay-btn");
  b.hidden = !button;
  if (button) b.textContent = button;
  overlay.hidden = false;
}

// ---------------------------------------------------------------- formatting

const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const sup = (n: number) => String(n).split("").map((c) => SUP[c] ?? c).join("");
function sci(x: number, digits = 2): string {
  if (x === 0 || !isFinite(x)) return String(x);
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e >= -2 && e < 4) return x.toPrecision(digits + 1).replace(/\.?0+$/, (m) => (m.includes(".") ? "" : m));
  const m = x / 10 ** e;
  return `${m.toFixed(digits - 1)}×10${sup(e)}`;
}
function duration(s: number): string {
  const a = Math.abs(s);
  if (a < 1e-3) return `${sci(s * 1e6)} µs`;
  if (a < 1) return `${(s * 1e3).toFixed(1)} ms`;
  if (a < 120) return `${s.toFixed(2)} s`;
  if (a < 7200) return `${(s / 60).toFixed(1)} min`;
  if (a < 172800) return `${(s / 3600).toFixed(1)} h`;
  if (a < 3.156e7 * 2) return `${(s / 86400).toFixed(1)} d`;
  return `${sci(s / 3.156e7)} yr`;
}
function length(m: number): string {
  if (m < 1e3) return `${m.toFixed(0)} m`;
  if (m < 1e8) return `${sci(m / 1e3)} km`;
  if (m < 1e15) return `${sci(m / 1.495978707e11)} AU`;
  return `${sci(m / 9.4607e15)} ly`;
}
const roman = (n: number) => {
  const t: [number, string][] = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
  let s = "";
  for (const [v, r] of t) while (n >= v) { s += r; n -= v; }
  return s;
};

// ---------------------------------------------------------------- state

const S = {
  a: 0.9,
  logM: 9.813,
  logMdot: -4,
  rOut: 30,
  fov: 42,
  ev: 0,
  skyGain: 36,
  logWarp: 0.7,
  logThrust: -2,
  tol: 1e-5,
  paused: false,
  toggles: { lensing: true, redshift: true, doppler: true, disk: true, stars: true, turb: true, luminet: false },
  plate: 1,
};
const M = () => 10 ** S.logM;
const accelUnit = () => SI.c ** 2 / rg(M()); // m/s^2 per geometric 1/M

const ob = new Observer({ a: S.a, mode: "static", r: 55, theta: (81 * Math.PI) / 180, prograde: true, gyro: false });
let trail: [number, number, number][] = [];
let diskIn = isco(S.a);
let tPeak = 0;

function diskTable() {
  diskIn = isco(S.a);
  const T = new Float32Array(DISK_N);
  tPeak = 0;
  for (let i = 0; i < DISK_N; i++) {
    const r = diskIn * (S.rOut / diskIn) ** (i / (DISK_N - 1));
    T[i] = diskTemperature(S.a, r, M(), 10 ** S.logMdot);
    tPeak = Math.max(tPeak, T[i]);
  }
  return T;
}

// ---------------------------------------------------------------- controls

const num = (id: string) => Number($<HTMLInputElement>(id).value);
const setOut = (id: string, text: string) => { $(`${id}-out`).textContent = text; };
const radioValue = (name: string) => (document.querySelector(`input[name="${name}"]:checked`) as HTMLInputElement).value;
const setRadio = (name: string, value: string) => { (document.querySelector(`input[name="${name}"][value="${value}"]`) as HTMLInputElement).checked = true; };

let renderer: Renderer | null = null;
let dirty = true; // anything that invalidates progressive accumulation

function minRadius() {
  const p = ob.p;
  return Observer.minRadius(S.a, p.mode, p.theta, p.prograde);
}

function syncObserverUI() {
  const p = ob.p;
  setRadio("mode", p.mode);
  const free = p.mode === "free";
  $<HTMLInputElement>("radius").disabled = free;
  $<HTMLInputElement>("theta").disabled = free || p.mode === "orbit";
  if (!free) {
    $<HTMLInputElement>("radius").value = String(Math.log10(p.r));
    $<HTMLInputElement>("theta").value = String((p.theta * 180) / Math.PI);
  }
  setOut("radius", free ? "free" : `${p.r.toFixed(p.r < 10 ? 3 : 1)} M`);
  setOut("theta", p.mode === "orbit" ? "90° (equator)" : `${((p.theta * 180) / Math.PI).toFixed(1)}°`);
  const notes: Record<Mode, string> = {
    static: "Rockets hold you fixed against the distant stars. Not possible inside the ergosphere.",
    zamo: `Zero angular momentum: frame dragging carries you round at ω = ${zamoOmega(S.a, position(S.a, p.r, p.theta, 0)).toPrecision(3)} M⁻¹.`,
    orbit: p.r < isco(S.a, p.prograde)
      ? `A circular geodesic, but unstable: you are inside the ISCO (${isco(S.a, p.prograde).toFixed(2)} M).`
      : `A circular geodesic with engines off. Stable outside the ISCO at ${isco(S.a, p.prograde).toFixed(2)} M.`,
    free: "Geodesic motion. W/S, A/D, R/F fire the rocket along the camera axes.",
  };
  $("mode-note").textContent = notes[p.mode];
}

function setMode(mode: Mode) {
  if (mode === "free") {
    ob.release();
    trail = [];
  } else {
    const wasFree = ob.p.mode === "free";
    ob.p.mode = mode;
    if (wasFree) {
      ob.p.r = ob.rNow;
      ob.p.theta = Math.acos(Math.max(-1, Math.min(1, ob.pos[3] / ob.rNow)));
      ob.phi = Math.atan2(ob.pos[2], ob.pos[1]) - Math.atan2(S.a, ob.p.r);
    }
    if (ob.p.r < minRadius()) {
      if (mode === "static" && ob.p.r > horizonOuter(S.a)) ob.p.mode = "zamo";
      else ob.p.r = Math.max(30, minRadius());
    }
    ob.p.r = Math.max(ob.p.r, minRadius());
    ob.ended = null;
    ob.place();
    trail = [];
    overlay.hidden = true;
  }
  dirty = true;
  syncObserverUI();
}

function updateHoleOutputs() {
  setOut("spin", `${S.a >= 0 ? "+" : "−"}${Math.abs(S.a).toFixed(3)}`);
  setOut("mass", `${sci(M())} M☉`);
  setOut("mdot", `${sci(10 ** S.logMdot)}`);
  setOut("rout", `${S.rOut.toFixed(1)} M`);
}

function updateTimeOutputs() {
  const w = 10 ** S.logWarp;
  setOut("warp", `${sci(w)} M/s · ${duration(w * tg(M()))}/s`);
  const th = 10 ** S.logThrust;
  setOut("thrust", `${sci(th)} M⁻¹ · ${sci((th * accelUnit()) / SI.g0)} g`);
}

function bind() {
  const onHole = () => {
    S.a = num("spin");
    S.logM = num("mass");
    S.logMdot = num("mdot");
    S.rOut = num("rout");
    updateHoleOutputs();
    updateTimeOutputs();
    renderer?.setDisk(diskTable());
    if (ob.p.a !== S.a) {
      ob.p.a = S.a;
      if (ob.p.mode === "free") setMode(Math.abs(S.a) > 0 ? "zamo" : "static");
      else setMode(ob.p.mode);
    }
    dirty = true;
  };
  for (const id of ["spin", "mass", "mdot", "rout"]) $(id).addEventListener("input", onHole);

  document.querySelectorAll('input[name="mode"]').forEach((el) => el.addEventListener("change", () => setMode(radioValue("mode") as Mode)));
  $("radius").addEventListener("input", () => {
    ob.p.r = Math.max(10 ** num("radius"), minRadius());
    ob.place();
    dirty = true;
    syncObserverUI();
  });
  $("theta").addEventListener("input", () => {
    ob.p.theta = (num("theta") * Math.PI) / 180;
    ob.p.r = Math.max(ob.p.r, minRadius());
    ob.place();
    dirty = true;
    syncObserverUI();
  });
  document.querySelectorAll('input[name="sense"]').forEach((el) => el.addEventListener("change", () => {
    ob.p.prograde = radioValue("sense") === "pro";
    if (ob.p.mode === "orbit") setMode("orbit");
  }));
  document.querySelectorAll('input[name="frame"]').forEach((el) => el.addEventListener("change", () => {
    ob.p.gyro = radioValue("frame") === "gyro";
    dirty = true;
  }));
  document.querySelectorAll('input[name="quality"]').forEach((el) => el.addEventListener("change", () => {
    S.tol = Number(radioValue("quality"));
    dirty = true;
  }));
  $("thrust").addEventListener("input", () => { S.logThrust = num("thrust"); updateTimeOutputs(); });
  $("warp").addEventListener("input", () => { S.logWarp = num("warp"); updateTimeOutputs(); });
  const onOptics = () => {
    S.fov = num("fov");
    S.ev = num("ev");
    S.skyGain = num("skygain");
    setOut("fov", `${S.fov.toFixed(0)}°`);
    setOut("ev", `${S.ev >= 0 ? "+" : "−"}${Math.abs(S.ev).toFixed(1)} EV`);
    setOut("skygain", `+${S.skyGain.toFixed(1)} EV`);
    dirty = true;
  };
  for (const id of ["fov", "ev", "skygain"]) $(id).addEventListener("input", onOptics);
  onOptics();
  for (const k of Object.keys(S.toggles) as (keyof typeof S.toggles)[]) {
    $(`t-${k}`).addEventListener("change", () => {
      S.toggles[k] = $<HTMLInputElement>(`t-${k}`).checked;
      dirty = true;
    });
  }
  $("release").addEventListener("click", () => setMode("free"));
  $("station").addEventListener("click", () => setMode(ob.rNow > ergosphere(S.a, ob.pos[3] / ob.rNow) + 0.01 ? "static" : "zamo"));
  $("overlay-btn").addEventListener("click", () => setMode("static"));
  $("pause").addEventListener("click", togglePause);
  updateHoleOutputs();
  updateTimeOutputs();
  syncObserverUI();
}

function togglePause() {
  S.paused = !S.paused;
  $("pause").setAttribute("aria-pressed", String(S.paused));
  $("pause").textContent = S.paused ? "Expose" : "Pause";
  if (S.paused) S.plate++;
}

// ---------------------------------------------------------------- input

const keys = new Set<string>();
let dragging = false;
canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
  canvas.focus();
});
canvas.addEventListener("pointerup", () => { dragging = false; });
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const k = ((S.fov * Math.PI) / 180) / canvas.clientHeight;
  ob.rotate(-e.movementX * k, e.movementY * k);
  dirty = true;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  S.fov = Math.min(150, Math.max(5, S.fov * Math.exp(e.deltaY * 0.001)));
  $<HTMLInputElement>("fov").value = String(S.fov);
  setOut("fov", `${S.fov.toFixed(0)}°`);
  dirty = true;
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT" && (e.target as HTMLInputElement).type !== "radio") return;
  if (e.code === "Space") { e.preventDefault(); togglePause(); return; }
  keys.add(e.code);
});
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());

function applyKeys(dt: number) {
  const k = (c: string) => (keys.has(c) ? 1 : 0);
  const fwd = k("KeyW") - k("KeyS"), right = k("KeyD") - k("KeyA"), up = k("KeyR") - k("KeyF");
  if (ob.p.mode === "free") {
    const th = 10 ** S.logThrust;
    ob.thrust = [fwd * th, right * th, up * th];
    return;
  }
  ob.thrust = [0, 0, 0];
  if (!fwd && !right && !up) return;
  ob.p.r = Math.max(minRadius(), Math.min(1000, ob.p.r * Math.exp(-fwd * dt * 0.6)));
  if (ob.p.mode !== "orbit") ob.p.theta = Math.min(Math.PI - 0.02, Math.max(0.02, ob.p.theta - up * dt * 0.5));
  ob.phi += right * dt * 0.5;
  ob.p.r = Math.max(ob.p.r, minRadius());
  ob.place();
  dirty = true;
  syncObserverUI();
}

// ---------------------------------------------------------------- readouts

let fpsAvg = 60;
let scale = 0.75;
let samples = 0;

function readouts() {
  const R = ob.readouts();
  const rgm = rg(M());
  const set = (id: string, v: string) => { $(id).textContent = v; };
  set("o-r", `${R.r.toFixed(R.r < 10 ? 4 : 2)} M`);
  set("o-rphys", length(R.r * rgm));
  set("o-theta", `${((R.theta * 180) / Math.PI).toFixed(1)}°`);
  const rp = horizonOuter(S.a);
  const region = R.r < horizonInner(S.a) ? "beyond the inner horizon" : R.inside ? "inside the event horizon" : R.r < ergosphere(S.a, Math.cos(R.theta)) ? "in the ergosphere" : "outside the ergosphere";
  set("o-region", region);
  $("o-region").parentElement!.classList.toggle("warn", R.inside);
  set("o-tau", duration(R.tau * tg(M())));
  set("o-t", `τ = ${R.tau.toFixed(1)} M · t = ${R.t.toFixed(1)} M`);
  set("o-dtdt", R.dtaudt.toFixed(4));
  set("o-v", R.vZamo === null ? "—" : `${R.vZamo.toFixed(4)} c`);
  set("o-gamma", R.vZamo === null ? "no ZAMO inside r₊" : `γ = ${(1 / Math.sqrt(1 - R.vZamo ** 2)).toFixed(3)}`);
  set("o-acc", `${sci((R.accel * accelUnit()) / SI.g0)} g`);
  set("o-accnote", ob.p.mode === "free" ? "rocket thrust" : ob.p.mode === "orbit" ? "free orbit" : "to hold station");
  const stretch = -R.tidal[0] * (SI.c / rgm) ** 2 * 2;
  set("o-tidal", `${sci(stretch / SI.g0)} g`);
  set("o-gahead", `×${R.gAhead.toFixed(3)} ahead`);
  set("o-gbehind", `×${R.gBehind.toFixed(3)} behind (frequency)`);
  set("o-tpeak", `${sci(tPeak, 3)} K`);
  set("o-lum", `L = ${sci((10 ** S.logMdot * eddingtonLuminosity(M())) / 3.828e26)} L☉`);
  set("o-rplus", `${rp.toFixed(3)} / ${horizonInner(S.a).toFixed(3)}`);
  set("o-risco", `ISCO ${isco(S.a).toFixed(2)} · photon ${photonOrbit(S.a).toFixed(2)}–${photonOrbit(S.a, false).toFixed(2)}`);
  set("o-fps", `${fpsAvg.toFixed(0)} fps`);
  set("o-scale", S.paused ? `${samples} samples · full res` : `scale ${(scale * 100).toFixed(0)}%`);

  const modeName: Record<Mode, string> = { static: "a static observer", zamo: "a zero-angular-momentum observer", orbit: `an observer on a ${ob.p.prograde ? "prograde" : "retrograde"} circular orbit`, free: "a freely falling observer" };
  $("caption").innerHTML = `<b>Plate ${roman(S.plate)}.</b> Kerr black hole of spin a = ${S.a.toFixed(3)} M and mass ${sci(M())} M☉ (r<sub>g</sub> = ${length(rgm)}), accreting at ${sci(10 ** S.logMdot)} Ṁ<sub>Edd</sub>; seen by ${modeName[ob.p.mode]} at r = ${R.r.toFixed(2)} M, θ = ${((R.theta * 180) / Math.PI).toFixed(0)}°. Field ${S.fov.toFixed(0)}°.${R.inside ? " <em>Inside the event horizon, every future direction leads inward.</em>" : ""}`;
  $("plate-no").textContent = roman(S.plate);

  drawChart($<HTMLCanvasElement>("chart"), { a: S.a, diskIn, diskOut: S.rOut, r: R.r, theta: R.theta, phi: Math.atan2(ob.pos[2], ob.pos[1]), trail });
}

// ---------------------------------------------------------------- main loop

async function main() {
  bind();
  if (!navigator.gpu) {
    showOverlay("WebGPU required", "This instrument traces every pixel on the GPU through WebGPU. Please use a current Chrome, Edge or Safari, or Firefox 141 or later.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    showOverlay("No GPU adapter", "WebGPU is present but no adapter is available. Hardware acceleration may be disabled.");
    return;
  }
  const device = await adapter.requestDevice();
  device.lost.then((info) => showOverlay("The GPU was lost", `${info.message || "The device was reset."} Reload the page to continue.`));
  device.addEventListener("uncapturederror", (e) => {
    console.error(e.error.message);
    showOverlay("The GPU reported an error", `<code>${e.error.message.split("\n")[0].replace(/</g, "&lt;")}</code>`);
  });
  const format = navigator.gpu.getPreferredCanvasFormat();
  const ctx = canvas.getContext("webgpu")!;
  ctx.configure({ device, format, alphaMode: "opaque" });

  const [diffuse, stars] = await Promise.all([loadDiffuse("/sky-diffuse.bin"), loadStars("/stars.bin")]);
  renderer = new Renderer(device, { traceCode, postCode, canvasFormat: format, diffuse, stars });
  renderer.setDisk(diskTable());
  overlay.hidden = true;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    const cap = Math.sqrt(3.7e6 / (w * h));
    if (cap < 1) { w = Math.round(w * cap); h = Math.round(h * cap); }
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    renderer!.resize(canvas.width, canvas.height);
    dirty = true;
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  const sky = skyRotation();
  let last = performance.now(), lastReadout = 0, lastTrail = 0, lastScale = scale;
  let ended: string | null = null;
  const frame = (now: number) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    applyKeys(dt);
    if (!S.paused) {
      ob.advance(10 ** S.logWarp * dt);
      if (ob.p.mode === "free") dirty = true;
      if (ob.p.mode === "free" && now - lastTrail > 100) {
        lastTrail = now;
        const r = ob.rNow;
        trail.push([r, Math.acos(Math.max(-1, Math.min(1, ob.pos[3] / r))), Math.atan2(ob.pos[2], ob.pos[1])]);
        if (trail.length > 600) trail.shift();
      }
    }
    if (ob.ended && ob.ended !== ended) {
      const rm = horizonInner(S.a);
      const msg: Record<string, [string, string]> = {
        "inner horizon": ["The Cauchy horizon", `You reached the inner horizon r₋ = ${rm.toFixed(3)} M after a proper time of ${duration(ob.tau * tg(M()))}. Beyond it the Kerr solution stops being predictive (mass inflation), so the instrument stops here.`],
        singularity: ["The singularity", "In the Schwarzschild interior r = 0 is a moment of time, not a place. Tidal forces diverge as you approach it, and the instrument stops here."],
        escaped: ["Escaped", "You have left the vicinity of the hole and are heading out into the Galaxy."],
      };
      const [t, b] = msg[ob.ended];
      showOverlay(t, b, "Return to station");
    }
    ended = ob.ended;

    const cam = ob.camera();
    const r = ob.rNow;
    const still = S.paused && !dirty && !dragging;
    const sc = still ? 1 : scale;
    if (sc !== lastScale) samples = 0;
    lastScale = sc;
    samples = still ? samples + 1 : 0;
    dirty = false;
    const t = S.toggles;
    const flags = (t.redshift ? FLAGS.redshift : 0) | (t.doppler ? FLAGS.doppler : 0) | (t.disk ? FLAGS.disk : 0) | (t.stars && !t.luminet ? FLAGS.stars : 0) | (t.turb && !t.luminet ? FLAGS.turbulence : 0) | (t.luminet ? FLAGS.luminet : 0);
    const W = Math.max(1, Math.round(canvas.width * sc)), H = Math.max(1, Math.round(canvas.height * sc));
    const params: TraceParams = {
      pos: cam.pos, frame: cam.frame, sky, width: W, height: H,
      jitter: samples > 1 ? [Math.random() - 0.5, Math.random() - 0.5] : [0, 0],
      fovY: (S.fov * Math.PI) / 180, a: S.a, bend: t.lensing ? 1 : 0, diskIn, diskOut: S.rOut,
      rEscape: Math.max(1000, 2 * r), tol: S.tol, maxSteps: 3000, flags, skyGain: 2 ** S.skyGain, turbPeriod: 150,
      inside: r < horizonOuter(S.a),
    };
    if (!t.lensing) params.frame = ob.flatFrame(); // flat space: a static camera with the same orientation
    const enc = device.createCommandEncoder();
    renderer!.trace(enc, params, 1 / Math.max(1, samples));
    renderer!.post(enc, ctx.getCurrentTexture().createView(), {
      w: W, h: H, dt, evComp: S.ev, bloom: 0.045, grain: 0.035, vignette: 0.32, luminet: t.luminet, time: now / 1000,
    });
    device.queue.submit([enc.finish()]);

    if (!still) {
      fpsAvg += (1 / Math.max(dt, 1e-3) - fpsAvg) * 0.08;
      if (fpsAvg < 45) scale = Math.max(0.3, scale * 0.97);
      else if (fpsAvg > 57) scale = Math.min(1, scale * 1.01);
    }
    if (now - lastReadout > 150) {
      lastReadout = now;
      readouts();
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  showOverlay("The instrument failed to start", String(e?.message ?? e));
});
