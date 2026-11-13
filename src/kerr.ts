// Kerr spacetime in double precision: the reference physics used by the UI, the observer worldline and the
// tests (the WGSL ray tracer is a single-precision port of rayRhs/escapeDirection below).
//
// Units G = c = M = 1. Ingoing Kerr–Schild Cartesian coordinates x^mu = (t, x, y, z), spin a along +z:
//   g_mn = eta_mn + f l_m l_n,   f = 2 r^3 / (r^4 + a^2 z^2),
//   l_m  = (1, (r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2), z / r),
//   (x^2 + y^2)/(r^2 + a^2) + z^2/r^2 = 1.
// These coordinates are regular across both horizons, so observers (and rays) can cross r_+.
// r and theta = acos(z/r) coincide with Boyer–Lindquist r, theta; d/dt and d/dphi are the same Killing vectors.

export type V4 = [number, number, number, number];
export type M4 = V4[];

// ---------------------------------------------------------------- metric

export function ksRadius(a: number, x: number, y: number, z: number): number {
  const w = x * x + y * y + z * z - a * a;
  const s = Math.sqrt(w * w + 4 * a * a * z * z);
  const r2 = w >= 0 ? 0.5 * (w + s) : (2 * a * a * z * z) / (s - w || 1e-300); // no cancellation either way
  return Math.sqrt(r2);
}

export interface KS {
  r: number;
  f: number;
  l: [number, number, number]; // spatial covariant l_i (l_t = 1)
  dr: [number, number, number]; // d_i r
  df: [number, number, number]; // d_i f
  dl: [number, number, number][]; // dl[i][j] = d_i l_j
}

export function ks(a: number, x: number, y: number, z: number): KS {
  const r = ksRadius(a, x, y, z);
  const r2 = r * r, a2 = a * a, D = r2 + a2;
  const w = x * x + y * y + z * z - a2;
  const q = r * Math.sqrt(w * w + 4 * a2 * z * z); // = r (2 r^2 - R^2 + a^2)
  const dr: [number, number, number] = [(r2 * x) / q, (r2 * y) / q, (r2 * z + a2 * z) / q];
  const den = r2 * r2 + a2 * z * z;
  const f = (2 * r2 * r) / den;
  const df = [0, 1, 2].map((i) => f * ((3 * dr[i]) / r - (4 * r2 * r * dr[i] + (i === 2 ? 2 * a2 * z : 0)) / den)) as [number, number, number];
  const lx = (r * x + a * y) / D, ly = (r * y - a * x) / D, lz = z / r;
  const dl = [0, 1, 2].map((i) => [
    (dr[i] * x + (i === 0 ? r : 0) + (i === 1 ? a : 0) - 2 * r * dr[i] * lx) / D,
    (dr[i] * y + (i === 1 ? r : 0) - (i === 0 ? a : 0) - 2 * r * dr[i] * ly) / D,
    (i === 2 ? 1 / r : 0) - (z * dr[i]) / r2,
  ]) as [number, number, number][];
  return { r, f, l: [lx, ly, lz], dr, df, dl };
}

const ETA = [-1, 1, 1, 1];

/** Covariant metric g_mn at spatial point p. `bend` = 0 gives flat spacetime (teaching toggle). */
export function metric(a: number, p: V4, bend = 1): M4 {
  const { f, l } = ks(a, p[1], p[2], p[3]);
  const L = [1, ...l];
  return [0, 1, 2, 3].map((m) => [0, 1, 2, 3].map((n) => (m === n ? ETA[m] : 0) + bend * f * L[m] * L[n])) as M4;
}

export function inverseMetric(a: number, p: V4, bend = 1): M4 {
  const { f, l } = ks(a, p[1], p[2], p[3]);
  const L = [-1, ...l]; // l^m = eta^mn l_n
  return [0, 1, 2, 3].map((m) => [0, 1, 2, 3].map((n) => (m === n ? ETA[m] : 0) - bend * f * L[m] * L[n])) as M4;
}

/** dg[k][m][n] = d_k g_mn (k = 0..3, d_t = 0). */
export function metricDerivs(a: number, p: V4, bend = 1): M4[] {
  const { f, l, df, dl } = ks(a, p[1], p[2], p[3]);
  const L = [1, ...l];
  const zero = () => [0, 1, 2, 3].map(() => [0, 0, 0, 0]) as M4;
  const out = [zero()];
  for (let i = 0; i < 3; i++) {
    const dL = [0, ...dl[i]];
    out.push([0, 1, 2, 3].map((m) => [0, 1, 2, 3].map((n) => bend * (df[i] * L[m] * L[n] + f * (dL[m] * L[n] + L[m] * dL[n])))) as M4);
  }
  return out;
}

/** Christoffel symbols G[m][a][b] = Gamma^m_ab. */
export function christoffel(a: number, p: V4, bend = 1): number[][][] {
  const gi = inverseMetric(a, p, bend), dg = metricDerivs(a, p, bend);
  const G: number[][][] = [];
  for (let m = 0; m < 4; m++) {
    G.push([]);
    for (let al = 0; al < 4; al++) {
      G[m].push([]);
      for (let be = 0; be < 4; be++) {
        let s = 0;
        for (let n = 0; n < 4; n++) s += gi[m][n] * (dg[al][n][be] + dg[be][n][al] - dg[n][al][be]);
        G[m][al].push(0.5 * s);
      }
    }
  }
  return G;
}

export const dot = (g: M4, u: V4, v: V4) => {
  let s = 0;
  for (let m = 0; m < 4; m++) for (let n = 0; n < 4; n++) s += g[m][n] * u[m] * v[n];
  return s;
};
/** g_mn v^n (or g^mn v_n when given the inverse metric). */
export const lower = (g: M4, v: V4) => g.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]) as V4;

// ---------------------------------------------------------------- characteristic radii (Boyer–Lindquist r)

export const horizonOuter = (a: number) => 1 + Math.sqrt(1 - a * a);
export const horizonInner = (a: number) => 1 - Math.sqrt(1 - a * a);
export const ergosphere = (a: number, cosTheta: number) => 1 + Math.sqrt(1 - a * a * cosTheta * cosTheta);

/** Bardeen, Press & Teukolsky (1972) marginally stable orbit. */
export function isco(a: number, prograde = true): number {
  if (a < 0) return isco(-a, !prograde); // orbit sense is relative to +z; the formula wants a >= 0
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 + (prograde ? -1 : 1) * Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

/** Equatorial circular photon orbit. */
export const photonOrbit = (a: number, prograde = true) => 2 * (1 + Math.cos((2 / 3) * Math.acos(prograde ? -a : a)));

/** Equatorial circular geodesic (BPT 1972): angular velocity, specific energy & angular momentum, u^t. */
export function circularOrbit(a: number, r: number, prograde = true) {
  const s = prograde ? 1 : -1, sr = Math.sqrt(r), r32 = r * sr;
  const den = Math.pow(r, 0.75) * Math.sqrt(r32 - 3 * sr + 2 * s * a);
  return {
    omega: s / (r32 + s * a),
    E: (r32 - 2 * sr + s * a) / den,
    L: (s * (r * r - 2 * s * a * sr + a * a)) / den,
    ut: (r32 + s * a) / den,
  };
}

// ---------------------------------------------------------------- Novikov–Thorne disk (Page & Thorne 1974)

/**
 * Time-averaged flux emitted from one face of the disk, in units of Mdot / M^2, from the conservation-law
 * integral  F = -(Mdot / 4 pi sqrt(-g)) Omega_r / (E - Omega L)^2  int_{r_ms}^r (E - Omega L) L_r dr
 * (sqrt(-g) = r in equatorial (t, r, z, phi)). Integrated numerically with Gauss–Legendre panels.
 */
export function pageThorneFlux(a: number, r: number): number {
  const rms = isco(a);
  if (r <= rms) return 0;
  const orb = (x: number) => circularOrbit(a, x);
  const dL = (x: number) => {
    const h = 1e-5 * x;
    return (orb(x + h).L - orb(x - h).L) / (2 * h);
  };
  // substitution x = r_ms + s^2 clusters nodes near r_ms where the integrand varies fastest
  const N = 24;
  const smax = Math.sqrt(r - rms);
  let I = 0;
  for (let k = 0; k < N; k++) {
    const s0 = (smax * k) / N, s1 = (smax * (k + 1)) / N;
    for (const [xg, wg] of GL5) {
      const s = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * xg;
      const x = rms + s * s;
      const o = orb(x);
      I += wg * 0.5 * (s1 - s0) * (o.E - o.omega * o.L) * dL(x) * 2 * s;
    }
  }
  const o = orb(r);
  const dOmega = -1.5 * Math.sqrt(r) / ((Math.pow(r, 1.5) + a) ** 2);
  return (-dOmega / (o.E - o.omega * o.L) ** 2) * I / (4 * Math.PI * r);
}

const GL5: [number, number][] = [
  [0, 0.5688888888888889],
  [-0.5384693101056831, 0.4786286704993665],
  [0.5384693101056831, 0.4786286704993665],
  [-0.906179845938664, 0.2369268850560891],
  [0.906179845938664, 0.2369268850560891],
];

// ---------------------------------------------------------------- physical units

export const SI = {
  G: 6.6743e-11,
  c: 299792458,
  Msun: 1.98847e30,
  sigmaSB: 5.670374419e-8,
  h: 6.62607015e-34,
  kB: 1.380649e-23,
  mp: 1.67262192369e-27,
  sigmaT: 6.6524587321e-29,
  g0: 9.80665,
};
export const rg = (Msun: number) => (SI.G * Msun * SI.Msun) / SI.c ** 2; // [m]
export const tg = (Msun: number) => rg(Msun) / SI.c; // [s]
/** Efficiency 1 - E_isco: Mdot_Edd = L_Edd / (eta c^2), so lambda = L / L_Edd. */
export const efficiency = (a: number) => 1 - circularOrbit(a, isco(a)).E;
export const eddingtonLuminosity = (Msun: number) => (4 * Math.PI * SI.G * Msun * SI.Msun * SI.mp * SI.c) / SI.sigmaT;

/** Disk effective temperature [K] at radius r for mass (solar masses) and Eddington ratio. */
export function diskTemperature(a: number, r: number, Msun: number, eddRatio: number): number {
  const mdot = (eddRatio * eddingtonLuminosity(Msun)) / (efficiency(a) * SI.c ** 2);
  const F = (pageThorneFlux(a, r) * mdot * SI.c ** 2) / rg(Msun) ** 2;
  return Math.pow(F / SI.sigmaSB, 0.25);
}

// ---------------------------------------------------------------- geodesics

/**
 * Hamiltonian H = 1/2 g^mn p_m p_n, with g^mn = eta^mn - f l^m l^n:
 *   dx^m/dl = eta^mn p_n - f (l.p) l^m,    dp_i/dl = 1/2 (l.p)^2 d_i f + f (l.p) p_j d_i l_j.
 * State y = [t, x, y, z, p_t, p_x, p_y, p_z] with covariant p. `force` adds a covariant 4-force (proper
 * acceleration of a rocket, a_m) for timelike worldlines parametrised by proper time.
 */
export function geodesicRhs(a: number, y: number[], bend = 1, force?: V4): number[] {
  const k = ks(a, y[1], y[2], y[3]);
  const f = bend * k.f;
  const [pt, px, py, pz] = [y[4], y[5], y[6], y[7]];
  const lp = -pt + k.l[0] * px + k.l[1] * py + k.l[2] * pz;
  const out = [-pt + f * lp, px - f * lp * k.l[0], py - f * lp * k.l[1], pz - f * lp * k.l[2], 0, 0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const dlp = px * k.dl[i][0] + py * k.dl[i][1] + pz * k.dl[i][2];
    out[5 + i] = bend * (0.5 * lp * lp * k.df[i]) + f * lp * dlp;
  }
  if (force) for (let m = 0; m < 4; m++) out[4 + m] += force[m];
  return out;
}

// Dormand–Prince 5(4) tableau (autonomous system, so the c_i nodes are not needed).
const DP_A = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
const DP_B = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
const DP_E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]; // b5 - b4

/** One Dormand–Prince step; returns the 5th-order solution and an error-norm estimate (scaled by `scale`). */
export function dpStep(rhs: (y: number[]) => number[], y: number[], h: number, scale: (i: number) => number) {
  const n = y.length;
  const k: number[][] = [rhs(y)];
  for (let s = 1; s < 7; s++) {
    const ys = y.slice();
    for (let j = 0; j < s; j++) {
      const c = DP_A[s][j];
      if (c) for (let i = 0; i < n; i++) ys[i] += h * c * k[j][i];
    }
    k.push(rhs(ys));
  }
  const out = y.slice();
  let err = 0;
  for (let i = 0; i < n; i++) {
    let e = 0;
    for (let s = 0; s < 7; s++) {
      out[i] += h * DP_B[s] * k[s][i];
      e += h * DP_E[s] * k[s][i];
    }
    const sc = scale(i);
    if (sc > 0) err = Math.max(err, Math.abs(e) / sc);
  }
  return { y: out, err };
}

/** Adaptive DP45 integration from y until `stop(y)` is truthy or maxSteps. Returns the final state. */
export function integrate(
  rhs: (y: number[]) => number[],
  y0: number[],
  h0: number,
  tol: number,
  stop: (y: number[], yPrev: number[]) => boolean,
  scale: (y: number[], i: number) => number,
  maxSteps = 100000,
  hMax = Infinity,
) {
  let y = y0, h = h0, steps = 0;
  while (steps++ < maxSteps) {
    const st = dpStep(rhs, y, h, (i) => scale(y, i));
    const err = st.err / tol;
    if (err <= 1 || Math.abs(h) < 1e-12) {
      const prev = y;
      y = st.y;
      if (stop(y, prev)) break;
    }
    const fac = err === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * Math.pow(err, -0.2)));
    h = Math.sign(h) * Math.min(Math.abs(h * fac), hMax);
  }
  return { y, steps };
}

// ---------------------------------------------------------------- conserved quantities (null or timelike)

/** Energy E = -p_t, axial angular momentum L = p_phi = x p_y - y p_x, Carter constant Q (BL form). */
export function constants(a: number, y: number[]) {
  const E = -y[4], L = y[1] * y[6] - y[2] * y[5];
  // Carter: Q = p_theta^2 + cos^2(th) (a^2 (mu^2 - E^2) + L^2 / sin^2(th)), with mu^2 = -2H.
  // p_theta = p . d/dtheta where d/dtheta (at fixed KS r, phi) = (z cos(phi)..., ...) computed from
  // x + iy = (r + i a) e^{i phi} sin(th), z = r cos(th).
  const p: V4 = [y[0], y[1], y[2], y[3]];
  const r = ksRadius(a, p[1], p[2], p[3]);
  const cth = p[3] / r, sth = Math.sqrt(Math.max(0, 1 - cth * cth));
  const phi = Math.atan2(p[2], p[1]) - Math.atan2(a, r); // KS phi
  const dth = [Math.cos(phi) * r * cth - Math.sin(phi) * a * cth, Math.sin(phi) * r * cth + Math.cos(phi) * a * cth, -r * sth];
  const pth = y[5] * dth[0] + y[6] * dth[1] + y[7] * dth[2];
  const gi = inverseMetric(a, p);
  const pv: V4 = [y[4], y[5], y[6], y[7]];
  const mu2 = -dot(gi, pv, pv);
  const Q = pth * pth + cth * cth * (a * a * (mu2 - E * E) + (sth > 1e-12 ? (L * L) / (sth * sth) : 0));
  return { E, L, Q, mu2 };
}

// ---------------------------------------------------------------- rays from the camera

export type RayEnd =
  | { kind: "escape"; dir: [number, number, number]; g: number; steps: number }
  | { kind: "horizon"; steps: number }
  | { kind: "disk"; r: number; phi: number; g: number; dt: number; steps: number }
  | { kind: "maxsteps"; steps: number };

export interface RayOpts {
  a: number;
  bend?: number;
  rEscape: number;
  diskIn: number;
  diskOut: number;
  inside: boolean; // camera inside r_+
  tol?: number;
  maxSteps?: number;
}

/**
 * Asymptotic direction of a ray leaving at large r: remaining Schwarzschild bending from the current point to
 * infinity by exact quadrature (Kerr corrections are O(a M / r^2)). `pk` is the spatial covariant momentum
 * pointing along the direction of travel, E > 0 its energy.
 */
export function escapeDirection(x: number[], pk: number[], E: number, M = 1): [number, number, number] {
  const r = Math.hypot(x[0], x[1], x[2]);
  const xh = x.map((v) => v / r);
  const Lv = [x[1] * pk[2] - x[2] * pk[1], x[2] * pk[0] - x[0] * pk[2], x[0] * pk[1] - x[1] * pk[0]];
  const b = Math.hypot(Lv[0], Lv[1], Lv[2]) / E;
  // in-plane unit vector perpendicular to x-hat, along the motion: t = (L x x-hat) / |L|
  const Ln = Math.hypot(Lv[0], Lv[1], Lv[2]);
  if (Ln < 1e-14 * r * E) return xh as [number, number, number];
  const t = [(Lv[1] * xh[2] - Lv[2] * xh[1]) / Ln, (Lv[2] * xh[0] - Lv[0] * xh[2]) / Ln, (Lv[0] * xh[1] - Lv[1] * xh[0]) / Ln];
  // remaining polar angle: int_0^{chi0} cos(c) / sqrt(cos^2 c + 2 M sin^3 c / b) dc, sin(chi0) = b / r
  const chi0 = Math.asin(Math.min(1, b / r));
  let dphi = 0;
  const N = 8;
  for (let k = 0; k < N; k++) {
    for (const [xg, wg] of GL5) {
      const c = (chi0 * (k + 0.5 + 0.5 * xg)) / N;
      const s = Math.sin(c), co = Math.cos(c);
      dphi += ((wg * 0.5 * chi0) / N) * (co / Math.sqrt(co * co + (2 * M * s * s * s) / b));
    }
  }
  return [0, 1, 2].map((i) => Math.cos(dphi) * xh[i] + Math.sin(dphi) * t[i]) as [number, number, number];
}

/**
 * Trace a ray backwards from the camera. `k` is the covariant momentum of the physical (future-directed)
 * photon arriving at the camera, normalised so that its energy measured by the camera is 1.
 */
export function traceRay(pos: V4, k: V4, o: RayOpts): RayEnd {
  const bend = o.bend ?? 1, a = o.a, rp = horizonOuter(a);
  const back = (y: number[]) => geodesicRhs(a, y, bend).map((v) => -v); // affine parameter runs backwards
  const tol = o.tol ?? 1e-9;
  let y = [...pos, ...k];
  let steps = 0;
  let h = 0.05 * Math.max(1, ksRadius(a, pos[1], pos[2], pos[3]));
  const maxSteps = o.maxSteps ?? 20000;
  const scale = (yy: number[], i: number) => (i >= 1 && i <= 3 ? 1e-3 + ksRadius(a, yy[1], yy[2], yy[3]) : 0);
  while (steps < maxSteps) {
    steps++;
    const st = dpStep(back, y, h, (i) => scale(y, i));
    const err = st.err / tol;
    if (err > 1 && Math.abs(h) > 1e-10) {
      h *= Math.max(0.2, 0.9 * Math.pow(err, -0.2));
      continue;
    }
    const prev = y, hTaken = h;
    y = st.y;
    h *= err === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * Math.pow(err, -0.2)));
    const r = ksRadius(a, y[1], y[2], y[3]);
    h = Math.min(h, 0.5 * r + 0.5);
    // disk crossing: equatorial plane z = 0 between ISCO and outer edge
    if (prev[3] * y[3] <= 0 && prev[3] !== y[3]) {
      const yc = refineCrossing(back, prev, y, hTaken);
      const rc = ksRadius(a, yc[1], yc[2], yc[3]);
      if (rc >= o.diskIn && rc <= o.diskOut) {
        const orb = circularOrbit(a, rc);
        const ut = diskUt(a, yc, orb.omega, bend);
        const kphi = yc[1] * yc[6] - yc[2] * yc[5];
        const g = 1 / (-ut * (yc[4] + orb.omega * kphi));
        const phi = Math.atan2(yc[2], yc[1]);
        return { kind: "disk", r: rc, phi, g, dt: yc[0] - pos[0], steps };
      }
    }
    if (r > o.rEscape) {
      const vr = y[1] * (y[1] - prev[1]) + y[2] * (y[2] - prev[2]) + y[3] * (y[3] - prev[3]);
      if (vr > 0) return { kind: "escape", dir: escapeDirection([y[1], y[2], y[3]], [-y[5], -y[6], -y[7]], -y[4]), g: 1 / -y[4], steps };
    }
    const pn = Math.hypot(y[5], y[6], y[7]) / Math.abs(y[4]);
    if (pn > 1e4 || (bend === 0 && r < rp) || (!o.inside && r < rp * 0.98)) return { kind: "horizon", steps };
  }
  return { kind: "maxsteps", steps };
}

/** Locate z = 0 inside the step prev -> next (step size h) by Illinois regula falsi on the step fraction. */
export function refineCrossing(rhs: (y: number[]) => number[], prev: number[], next: number[], h: number) {
  let s0 = 0, z0 = prev[3], s1 = 1, z1 = next[3], y = next, side = 0;
  for (let it = 0; it < 6; it++) {
    const s = (s0 * z1 - s1 * z0) / (z1 - z0);
    y = dpStep(rhs, prev, s * h, () => 0).y;
    if (y[3] * z1 > 0) {
      s1 = s; z1 = y[3];
      if (side === -1) z0 *= 0.5;
      side = -1;
    } else {
      s0 = s; z0 = y[3];
      if (side === 1) z1 *= 0.5;
      side = 1;
    }
  }
  return y;
}

/** u^t of a circular equatorial emitter with angular velocity omega, from the metric (bend-aware). */
export function diskUt(a: number, p: number[], omega: number, bend = 1): number {
  const k = ks(a, p[1], p[2], p[3]);
  const lv = 1 + omega * (p[1] * k.l[1] - p[2] * k.l[0]);
  return 1 / Math.sqrt(1 - omega * omega * (p[1] * p[1] + p[2] * p[2]) - bend * k.f * lv * lv);
}
