// The observer: a timelike worldline in Kerr with an orthonormal tetrad (the camera), plus everything the
// instruments read off it. Double precision throughout.
import {
  christoffel, circularOrbit, dot, dpStep, geodesicRhs, horizonInner, horizonOuter, inverseMetric, ksRadius,
  lower, metric, type M4, type V4,
} from "./kerr.ts";

export type Mode = "static" | "zamo" | "orbit" | "free";

/** KS Cartesian position for Boyer–Lindquist-like (r, theta) and KS azimuth phi. */
export function position(a: number, r: number, theta: number, phi: number): V4 {
  const s = Math.sin(theta);
  return [0, (r * Math.cos(phi) - a * Math.sin(phi)) * s, (r * Math.sin(phi) + a * Math.cos(phi)) * s, r * Math.cos(theta)];
}

export const killing = (p: V4, omega: number): V4 => [1, -omega * p[2], omega * p[1], 0]; // d_t + omega d_phi

/** Normalised 4-velocity along d_t + omega d_phi, or null if that direction is not timelike here. */
export function stationary(a: number, p: V4, omega: number, bend = 1): V4 | null {
  const xi = killing(p, omega);
  const n = -dot(metric(a, p, bend), xi, xi);
  return n > 0 ? (xi.map((v) => v / Math.sqrt(n)) as V4) : null;
}

/** Frame-dragging angular velocity omega = -g_tphi / g_phiphi (zero angular momentum observers). */
export function zamoOmega(a: number, p: V4) {
  const g = metric(a, p);
  const ph: V4 = [0, -p[2], p[1], 0], t: V4 = [1, 0, 0, 0];
  return -dot(g, t, ph) / dot(g, ph, ph);
}

/** Gram–Schmidt: orthonormal spatial triad (forward, right, up) orthogonal to u, from coordinate hints. */
export function triad(g: M4, u: V4, hints: V4[]): V4[] {
  const out: V4[] = [];
  for (const h of hints) {
    let v = h.slice() as V4;
    const vu = dot(g, v, u);
    v = v.map((c, i) => c + vu * u[i]) as V4; // u.u = -1
    for (const e of out) {
      const ve = dot(g, v, e);
      v = v.map((c, i) => c - ve * e[i]) as V4;
    }
    const n = Math.sqrt(dot(g, v, v));
    out.push(v.map((c) => c / n) as V4);
  }
  return out;
}

const cross3 = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Reference triad pointing at the hole: forward = radially inward, up = spin axis, right = forward x up. */
export function trackingTriad(g: M4, u: V4, p: V4): V4[] {
  const R = Math.hypot(p[1], p[2], p[3]);
  const fwd = [-p[1] / R, -p[2] / R, -p[3] / R];
  let up = [0, 0, 1];
  if (Math.abs(fwd[2]) > 0.999) up = [0, p[3] > 0 ? 1 : -1, 0];
  const right = cross3(fwd, up);
  return triad(g, u, [[0, ...fwd] as V4, [0, ...right] as V4, [0, ...up] as V4]);
}

// ---------------------------------------------------------------- curvature

/** Riemann tensor R^r_{s m n} (numerical derivative of analytic Christoffels; error ~1e-8 relative). */
export function riemann(a: number, p: V4): number[][][][] {
  const G = christoffel(a, p);
  const h = 1e-4 * Math.max(1, ksRadius(a, p[1], p[2], p[3]));
  const dG: number[][][][] = [G.map((x) => x.map((y) => y.map(() => 0)))];
  for (let i = 1; i < 4; i++) {
    const pp = p.slice() as V4, pm = p.slice() as V4;
    pp[i] += h; pm[i] -= h;
    const Gp = christoffel(a, pp), Gm = christoffel(a, pm);
    dG.push(Gp.map((x, r) => x.map((y, s) => y.map((v, t) => (v - Gm[r][s][t]) / (2 * h)))));
  }
  const R: number[][][][] = [];
  for (let r = 0; r < 4; r++) {
    R.push([]);
    for (let s = 0; s < 4; s++) {
      R[r].push([]);
      for (let m = 0; m < 4; m++) {
        R[r][s].push([]);
        for (let n = 0; n < 4; n++) {
          let v = dG[m][r][n][s] - dG[n][r][m][s];
          for (let l = 0; l < 4; l++) v += G[r][m][l] * G[l][n][s] - G[r][n][l] * G[l][m][s];
          R[r][s][m].push(v);
        }
      }
    }
  }
  return R;
}

/** Electric (tidal) and magnetic parts of the Weyl = Riemann tensor in the frame (u, e1, e2, e3). */
export function tidalTensors(a: number, p: V4, frame: V4[]) {
  const g = metric(a, p), Rup = riemann(a, p);
  // R_{abcd} on frame vectors: R(A,B,C,D) = g_{mr} R^r_{snq} A^m B^s C^n D^q
  const Rf = (A: V4, B: V4, C: V4, D: V4) => {
    const Al = lower(g, A);
    let v = 0;
    for (let r = 0; r < 4; r++) for (let s = 0; s < 4; s++) for (let n = 0; n < 4; n++) for (let q = 0; q < 4; q++)
      v += Al[r] * Rup[r][s][n][q] * B[s] * C[n] * D[q];
    return v;
  };
  const [u, ...e] = frame;
  const E = [0, 1, 2].map((i) => [0, 1, 2].map((j) => Rf(e[i], u, e[j], u)));
  const B = [0, 1, 2].map((i) => [0, 1, 2].map((j) => {
    // B_ij = 1/2 eps_ipq R_pqj0
    const [p1, q1] = [(i + 1) % 3, (i + 2) % 3];
    return Rf(e[p1], e[q1], e[j], u);
  }));
  return { E, B };
}

/** Eigenvalues of a symmetric 3x3 matrix (trigonometric method), ascending. */
export function eigSym3(A: number[][]): number[] {
  const p1 = A[0][1] ** 2 + A[0][2] ** 2 + A[1][2] ** 2;
  const q = (A[0][0] + A[1][1] + A[2][2]) / 3;
  const p2 = (A[0][0] - q) ** 2 + (A[1][1] - q) ** 2 + (A[2][2] - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  if (p < 1e-300) return [q, q, q];
  const Bm = A.map((row, i) => row.map((v, j) => (v - (i === j ? q : 0)) / p));
  const det = Bm[0][0] * (Bm[1][1] * Bm[2][2] - Bm[1][2] * Bm[2][1]) - Bm[0][1] * (Bm[1][0] * Bm[2][2] - Bm[1][2] * Bm[2][0]) + Bm[0][2] * (Bm[1][0] * Bm[2][1] - Bm[1][1] * Bm[2][0]);
  const phi = Math.acos(Math.min(1, Math.max(-1, det / 2))) / 3;
  const e1 = q + 2 * p * Math.cos(phi), e3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  return [e3, 3 * q - e1 - e3, e1];
}

// ---------------------------------------------------------------- the observer

export interface ObserverParams {
  a: number;
  mode: Mode;
  r: number;
  theta: number; // polar angle from the spin axis (hover modes)
  prograde: boolean;
  gyro: boolean; // Fermi–Walker (gyroscope) frame instead of tracking the hole (free fall)
}

export interface Camera {
  pos: V4;
  frame: V4[]; // u, forward, right, up (contravariant components)
}

export class Observer {
  p: ObserverParams;
  pos: V4 = [0, 0, 0, 0];
  phi = 0;
  tau = 0;
  u: V4 = [1, 0, 0, 0];
  gyroTriad: V4[] | null = null; // FW-transported triad (free fall + gyro)
  look = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // user rotation of the triad (rows: camera f, r, u in triad basis)
  thrust = [0, 0, 0]; // proper acceleration along camera forward/right/up [1/M]
  ended: string | null = null;

  constructor(p: ObserverParams) {
    this.p = { ...p };
    this.place();
  }

  get rNow() {
    return ksRadius(this.p.a, this.pos[1], this.pos[2], this.pos[3]);
  }

  /** Re-seat the observer on its stationary worldline (hover / orbit modes). */
  place() {
    const { a, mode, r, theta } = this.p;
    if (mode === "free") return;
    this.ended = null;
    const th = mode === "orbit" ? Math.PI / 2 : theta;
    const t = this.pos[0];
    this.pos = position(a, r, th, this.phi);
    this.pos[0] = t;
    this.u = this.stationaryU() ?? this.u;
  }

  omega(): number {
    const { a, mode, r, prograde } = this.p;
    if (mode === "orbit") return circularOrbit(a, r, prograde).omega;
    if (mode === "zamo") return zamoOmega(a, this.pos);
    return 0;
  }

  stationaryU(): V4 | null {
    return stationary(this.p.a, this.pos, this.omega());
  }

  /** Bounds of the radius slider for the current mode (stationary observers need a timelike Killing orbit). */
  static minRadius(a: number, mode: Mode, theta: number, prograde: boolean) {
    if (mode === "static") return 1 + Math.sqrt(1 - a * a * Math.cos(theta) ** 2) + 1e-3;
    if (mode === "zamo") return horizonOuter(a) + 1e-3;
    if (mode === "orbit") return 2 * (1 + Math.cos((2 / 3) * Math.acos(prograde ? -a : a))) + 1e-2;
    return 0;
  }

  release() {
    if (this.p.mode === "free") return;
    this.gyroTriad = this.referenceTriad();
    this.p.mode = "free";
  }

  referenceTriad(): V4[] {
    const g = metric(this.p.a, this.pos);
    if (this.p.mode === "free" && this.p.gyro && this.gyroTriad) return triad(g, this.u, this.gyroTriad);
    return trackingTriad(g, this.u, this.pos);
  }

  private applyLook(tri: V4[]): V4[] {
    return this.look.map((row) => [0, 1, 2, 3].map((m) => row[0] * tri[0][m] + row[1] * tri[1][m] + row[2] * tri[2][m]) as V4);
  }

  camera(): Camera {
    return { pos: this.pos, frame: [this.u, ...this.applyLook(this.referenceTriad())] };
  }

  /** Camera frame of a static observer in flat spacetime (light bending switched off), same orientation. */
  flatFrame(): V4[] {
    const eta: M4 = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    const u: V4 = [1, 0, 0, 0];
    const tri = this.p.mode === "free" && this.p.gyro && this.gyroTriad ? triad(eta, u, this.gyroTriad) : trackingTriad(eta, u, this.pos);
    return [u, ...this.applyLook(tri)];
  }

  /** Rotate the view: yaw about camera up, pitch about camera right (radians). */
  rotate(yaw: number, pitch: number) {
    const rot = (v: number[], w: number[], ang: number) => v.map((x, i) => Math.cos(ang) * x + Math.sin(ang) * w[i]);
    const [f, r, u] = this.look;
    const f1 = rot(f, r, yaw);
    const f2 = rot(f1, u, pitch), u2 = rot(u, f1, -pitch);
    const n = (v: number[]) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
    const F = n(f2), U = n(u2.map((x, i) => x - (u2[0] * F[0] + u2[1] * F[1] + u2[2] * F[2]) * F[i]));
    // right = forward x up in space; (forward, right, up) is a left-handed basis, hence the swapped cross product
    this.look = [F, cross3(U, F), U];
  }

  /** Advance proper time by dtau (geometric units). */
  advance(dtau: number) {
    if (this.ended || dtau <= 0) return;
    const { a, mode } = this.p;
    if (mode !== "free") {
      const u = this.stationaryU();
      if (!u) return;
      this.u = u;
      const dt = u[0] * dtau;
      this.phi += this.omega() * dt;
      const t = this.pos[0] + dt;
      this.place();
      this.pos[0] = t;
      this.tau += dtau;
      return;
    }
    this.freeFall(dtau);
    const r = this.rNow;
    const rEnd = Math.max(horizonInner(a), 0) + 0.02;
    if (r < rEnd) this.ended = a > 0 ? "inner horizon" : "singularity";
    else if (r > 5000) this.ended = "escaped";
  }

  /** Geodesic (or rocket) motion with the tetrad Fermi–Walker transported along. */
  freeFall(dtau: number) {
    const a = this.p.a;
    const cam = this.camera();
    const tri0 = this.gyroTriad ?? trackingTriad(metric(a, this.pos), this.u, this.pos);
    // thrust is given in the camera frame
    const acc = [0, 1, 2, 3].map((m) => this.thrust[0] * cam.frame[1][m] + this.thrust[1] * cam.frame[2][m] + this.thrust[2] * cam.frame[3][m]) as V4;
    const accTri = tri0.map((e) => dot(metric(a, this.pos), acc, e)); // thrust components on the FW triad (constant)
    const pack = (pos: V4, ucov: V4, tri: V4[]) => [...pos, ...ucov, ...tri.flat()];
    const rhs = (y: number[]) => {
      const p = y.slice(0, 4) as V4;
      const g = metric(a, p), gi = inverseMetric(a, p);
      const ucov = y.slice(4, 8) as V4;
      const u = lower(gi, ucov);
      const tri = [0, 1, 2].map((i) => y.slice(8 + 4 * i, 12 + 4 * i) as V4);
      const A = [0, 1, 2, 3].map((m) => accTri[0] * tri[0][m] + accTri[1] * tri[1][m] + accTri[2] * tri[2][m]) as V4;
      const out = geodesicRhs(a, y.slice(0, 8), 1, lower(g, A));
      const G = christoffel(a, p);
      for (let i = 0; i < 3; i++) {
        const e = tri[i];
        const ae = dot(g, A, e);
        for (let m = 0; m < 4; m++) {
          let v = ae * u[m];
          for (let al = 0; al < 4; al++) for (let be = 0; be < 4; be++) v -= G[m][al][be] * u[al] * e[be];
          out.push(v);
        }
      }
      return out;
    };
    let y = pack(this.pos, lower(metric(a, this.pos), this.u), tri0);
    let left = dtau, h = Math.min(dtau, 0.02 * Math.max(0.1, this.rNow));
    while (left > 1e-12) {
      h = Math.min(h, left);
      const st = dpStep(rhs, y, h, (i) => (i < 4 ? 1e-3 + this.rNow : 1));
      const err = st.err / 1e-10;
      if (err <= 1 || h < 1e-9) {
        y = st.y;
        left -= h;
      }
      h *= Math.min(4, Math.max(0.2, 0.9 * Math.pow(Math.max(err, 1e-12), -0.2)));
    }
    this.pos = y.slice(0, 4) as V4;
    const g = metric(a, this.pos);
    const u = lower(inverseMetric(a, this.pos), y.slice(4, 8) as V4);
    const n = Math.sqrt(-dot(g, u, u)); // renormalise against round-off
    this.u = u.map((c) => c / n) as V4;
    this.gyroTriad = triad(g, this.u, [0, 1, 2].map((i) => y.slice(8 + 4 * i, 12 + 4 * i) as V4));
    this.tau += dtau;
  }

  /** Instrument readouts (geometric units; the UI converts). */
  readouts() {
    const a = this.p.a, p = this.pos, u = this.u, g = metric(a, p);
    const r = this.rNow;
    const theta = Math.acos(Math.max(-1, Math.min(1, p[3] / r)));
    const cam = this.camera();
    // proper acceleration needed to stay on this worldline
    let accel = Math.hypot(...this.thrust);
    if (this.p.mode !== "free") {
      const om = this.omega();
      const V = (q: V4) => { const xi = killing(q, om); return 0.5 * Math.log(-dot(metric(a, q), xi, xi)); };
      const h = 1e-5 * Math.max(1, r);
      const ai = [1, 2, 3].map((i) => {
        const qp = p.slice() as V4, qm = p.slice() as V4;
        qp[i] += h; qm[i] -= h;
        return (V(qp) - V(qm)) / (2 * h);
      });
      const gi = inverseMetric(a, p);
      let s = 0;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += gi[i + 1][j + 1] * ai[i] * ai[j];
      accel = Math.sqrt(Math.max(0, s));
    }
    // speed relative to the local ZAMO (defined outside the horizon)
    let vZamo: number | null = null;
    if (r > horizonOuter(a) + 1e-6) {
      const z = stationary(a, p, zamoOmega(a, p));
      if (z) vZamo = Math.sqrt(Math.max(0, 1 - 1 / dot(g, u, z) ** 2));
    }
    const { E } = tidalTensors(a, p, cam.frame);
    const tidal = eigSym3(E);
    // frequency shift of starlight arriving from straight ahead: g = 1 / (-k_t), k = u - forward
    const k = [0, 1, 2, 3].map((m) => u[m] - cam.frame[1][m]) as V4;
    const gAhead = 1 / -lower(g, k)[0];
    const kz = [0, 1, 2, 3].map((m) => u[m] + cam.frame[1][m]) as V4;
    const gBehind = 1 / -lower(g, kz)[0];
    return { r, theta, t: p[0], tau: this.tau, dtaudt: 1 / u[0], vZamo, accel, tidal, gAhead, gBehind, inside: r < horizonOuter(a) };
  }
}
