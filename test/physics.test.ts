// Physics verification against independent analytic results. Run: npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  circularOrbit, constants, diskTemperature, dot, efficiency, escapeDirection, geodesicRhs, horizonOuter, integrate,
  inverseMetric, isco, ks, ksRadius, lower, metric, pageThorneFlux, photonOrbit, traceRay, type V4,
} from "../src/kerr.ts";
import { Observer, eigSym3, position, stationary, tidalTensors, trackingTriad, triad } from "../src/observer.ts";
import { blackbodyXYZ } from "../src/color.ts";

const close = (got: number, want: number, tol: number, msg = "") =>
  assert.ok(Math.abs(got - want) <= tol, `${msg} got ${got}, want ${want} (tol ${tol})`);

/** Camera at rest (static or given 4-velocity) and a ray looking along (forward, right, up) components n. */
function cameraRay(a: number, pos: V4, n: number[], u?: V4) {
  const g = metric(a, pos);
  const uu = u ?? stationary(a, pos, 0)!;
  const [f, r, up] = trackingTriad(g, uu, pos);
  const k = [0, 1, 2, 3].map((m) => uu[m] - n[0] * f[m] - n[1] * r[m] - n[2] * up[m]) as V4;
  return { k: lower(g, k), u: uu };
}

const rayOpts = (a: number, inside = false) => ({ a, rEscape: 2000, diskIn: 1e9, diskOut: 0, inside, tol: 1e-11, maxSteps: 40000 });

test("metric: inverse, null congruence, horizon radius", () => {
  for (const a of [0, 0.5, 0.998]) {
    const p: V4 = [0, 3.1, -1.7, 2.2];
    const g = metric(a, p), gi = inverseMetric(a, p);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += g[i][k] * gi[k][j];
      close(s, i === j ? 1 : 0, 1e-13);
    }
    const { l } = ks(a, p[1], p[2], p[3]);
    close(dot(gi, [1, ...l] as V4, [1, ...l] as V4), 0, 1e-13, "l null");
    // r_+ is where g^rr ~ Delta = r^2 - 2r + a^2 vanishes
    const rp = horizonOuter(a);
    close(rp * rp - 2 * rp + a * a, 0, 1e-14);
    close(ksRadius(a, ...(position(a, 2.345, 0.7, 1.1).slice(1) as [number, number, number])), 2.345, 1e-13);
  }
});

test("ISCO (BPT 1972) is the minimum of the circular-orbit energy; photon orbits", () => {
  close(isco(0), 6, 1e-12);
  close(isco(1), 1, 1e-6);
  close(isco(1, false), 9, 1e-12);
  for (const a of [0, 0.3, 0.9, 0.99]) {
    const r = isco(a), h = 1e-4;
    const dE = (circularOrbit(a, r + h).E - circularOrbit(a, r - h).E) / (2 * h);
    close(dE, 0, 1e-7, `dE/dr at ISCO a=${a}`);
    // photon orbit: where circular timelike orbits become null (E diverges)
    const rph = photonOrbit(a);
    close(rph ** 1.5 - 3 * Math.sqrt(rph) + 2 * a, 0, 1e-12, `photon orbit a=${a}`);
  }
  close(photonOrbit(0), 3, 1e-12);
  close(efficiency(0), 1 - Math.sqrt(8 / 9), 1e-12);
});

function criticalAngle(a: number, pos: V4, side: number) {
  let lo = 0.0, hi = 0.5; // angle from forward (toward the hole): lo captured, hi escapes
  for (let i = 0; i < 40; i++) {
    const m = 0.5 * (lo + hi);
    const { k } = cameraRay(a, pos, [Math.cos(m), side * Math.sin(m), 0]);
    const end = traceRay(pos, k, rayOpts(a));
    if (end.kind === "escape") hi = m;
    else lo = m;
  }
  const { k } = cameraRay(a, pos, [Math.cos(lo), side * Math.sin(lo), 0]);
  return k;
}

test("Schwarzschild shadow: critical impact parameter 3 sqrt(3) M", () => {
  const pos = position(0, 500, Math.PI / 2, 0);
  const k = criticalAngle(0, pos, 1);
  const { E, L } = constants(0, [...pos, ...k]);
  close(Math.abs(L / E), 3 * Math.sqrt(3), 2e-7);
});

test("Kerr equatorial shadow edges match Bardeen's critical photon orbits", () => {
  const a = 0.9;
  const xi = (r: number) => -(r ** 3 - 3 * r * r + a * a * r + a * a) / (a * (r - 1));
  const pos = position(a, 500, Math.PI / 2, 0);
  for (const side of [1, -1]) {
    const k = criticalAngle(a, pos, side);
    const { E, L } = constants(a, [...pos, ...k]);
    close(L / E, xi(photonOrbit(a, L > 0)), 2e-6, `side ${side}`);
  }
});

test("Kerr capture vs. Carter radial potential for random rays", () => {
  const a = 0.95, rp = horizonOuter(a);
  const pos = position(a, 12, 1.0, 0.3);
  const u = stationary(a, pos, 0)!;
  let rng = 12345;
  const rand = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
  let checked = 0;
  for (let n = 0; n < 300; n++) {
    const d = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
    const l = Math.hypot(...d);
    const { k } = cameraRay(a, pos, d.map((x) => x / l), u);
    const y = [...pos, ...k];
    const { E, L, Q } = constants(a, y);
    const lam = L / E, eta = Q / (E * E);
    const R = (r: number) => (r * r + a * a - a * lam) ** 2 - (r * r - 2 * r + a * a) * (eta + (lam - a) ** 2);
    // radial direction of the backward ray
    const kk = ks(a, y[1], y[2], y[3]);
    const dx = geodesicRhs(a, y).map((v) => -v);
    const vr = kk.dr[0] * dx[1] + kk.dr[1] * dx[2] + kk.dr[2] * dx[3];
    const rc = 12;
    const grid = (r0: number, r1: number) => Array.from({ length: 4000 }, (_, i) => r0 * Math.pow(r1 / r0, (i + 0.5) / 4000));
    const minIn = Math.min(...grid(rp * 1.0001, rc * 0.9999).map((r) => R(r) / r ** 4));
    const minOut = Math.min(...grid(rc * 1.0001, 1e5).map((r) => R(r) / r ** 4));
    if (Math.abs(minIn) < 1e-3 || Math.abs(minOut) < 1e-3) continue; // near-critical: skip
    const captured = vr < 0 ? minIn > 0 : minOut < 0;
    const end = traceRay(pos, k, rayOpts(a));
    assert.equal(end.kind === "horizon", captured, `ray ${n}: lambda ${lam} eta ${eta} ended ${end.kind}`);
    checked++;
  }
  assert.ok(checked > 200, `checked ${checked}`);
});

test("conservation of E, L_z, Carter Q and H along strong-field rays", () => {
  const a = 0.9;
  const pos = position(a, 8, 1.2, 0);
  for (const d of [[0.9, 0.3, 0.2], [0.95, -0.2, 0.25], [0.6, 0.6, -0.5]]) {
    const l = Math.hypot(...d);
    const { k } = cameraRay(a, pos, d.map((x) => x / l));
    const y0 = [...pos, ...k];
    const c0 = constants(a, y0);
    const back = (y: number[]) => geodesicRhs(a, y).map((v) => -v);
    const { y } = integrate(back, y0, 0.01, 1e-12, (yy) => ksRadius(a, yy[1], yy[2], yy[3]) > 200 || ksRadius(a, yy[1], yy[2], yy[3]) < 1.5,
      (yy, i) => (i >= 1 && i <= 3 ? 1 + ksRadius(a, yy[1], yy[2], yy[3]) : 0));
    const c1 = constants(a, y);
    close(c1.E, c0.E, 1e-11, "E");
    close(c1.L, c0.L, 1e-10, "L");
    close(c1.Q, c0.Q, 1e-8 * (1 + Math.abs(c0.Q)), "Q");
    close(c1.mu2, 0, 1e-8, "null");
  }
});

test("weak-field deflection 4M/b + 15 pi M^2/4b^2 + ... (escapeDirection + integrator)", () => {
  for (const b of [200, 1000]) {
    const p: V4 = [0, 0, b, 0];
    // null covector with k_t = -1 travelling along +x
    const gi = inverseMetric(0, p);
    const A = gi[1][1], B = 2 * gi[0][1] * -1, C = gi[0][0];
    const kx = (-B + Math.sqrt(B * B - 4 * A * C)) / (2 * A);
    const y0 = [...p, -1, kx, 0, 0];
    const L = constants(0, y0).L;
    const bb = Math.abs(L);
    const run = (sign: number) => integrate((y) => geodesicRhs(0, y).map((v) => sign * v), y0, 1, 1e-13,
      (y) => Math.hypot(y[1], y[2], y[3]) > 1e5, (y, i) => (i >= 1 && i <= 3 ? 1 + Math.hypot(y[1], y[2], y[3]) : 0)).y;
    const yf = run(1), yb = run(-1);
    const dOut = escapeDirection([yf[1], yf[2], yf[3]], [yf[5], yf[6], yf[7]], 1);
    const dIn = escapeDirection([yb[1], yb[2], yb[3]], [-yb[5], -yb[6], -yb[7]], 1); // where it came from
    const ang = Math.acos(-(dOut[0] * dIn[0] + dOut[1] * dIn[1] + dOut[2] * dIn[2]));
    const m = 1 / bb;
    const want = 4 * m + (15 * Math.PI / 4) * m ** 2 + (128 / 3) * m ** 3 + (3465 * Math.PI / 64) * m ** 4;
    close(ang, want, 2e-8, `b=${b}`);
  }
});

test("disk redshift seen from the axis: g = sqrt(1 - 3M/r) / sqrt(1 - 2M/D)", () => {
  const D = 100, pos = position(0, D, 0, 0);
  for (const ang of [0.08, 0.1, 0.15, 0.25]) {
    const { k } = cameraRay(0, pos, [Math.cos(ang), Math.sin(ang), 0]);
    const end = traceRay(pos, k, { ...rayOpts(0), diskIn: 6, diskOut: 1e3 });
    assert.equal(end.kind, "disk");
    if (end.kind === "disk") close(end.g, Math.sqrt(1 - 3 / end.r) / Math.sqrt(1 - 2 / D), 1e-9);
  }
});

test("Page-Thorne flux: Newtonian limit, closed form, and L = (1 - E_isco) Mdot", () => {
  for (const a of [0, 0.7, 0.998]) {
    // closed form (Page & Thorne 1974 in the form of e.g. Krolik 1999)
    const x0 = Math.sqrt(isco(a)), ac = Math.acos(a);
    const x1 = 2 * Math.cos(ac / 3 - Math.PI / 3), x2 = 2 * Math.cos(ac / 3 + Math.PI / 3), x3 = -2 * Math.cos(ac / 3);
    const term = (xi: number, xj: number, xk: number, x: number) =>
      Math.abs(xi) < 1e-9 ? 0 : (3 * (xi - a) ** 2) / (xi * (xi - xj) * (xi - xk)) * Math.log((x - xi) / (x0 - xi));
    const Fcf = (r: number) => {
      const x = Math.sqrt(r);
      const br = x - x0 - 1.5 * a * Math.log(x / x0) - term(x1, x2, x3, x) - term(x2, x1, x3, x) - term(x3, x1, x2, x);
      return (3 / (8 * Math.PI)) * br / (x ** 4 * (x ** 3 - 3 * x + 2 * a));
    };
    for (const r of [isco(a) * 1.01, 3, 7, 20, 300]) {
      if (r <= isco(a)) continue;
      close(pageThorneFlux(a, r) / Fcf(r), 1, 1e-6, `closed form a=${a} r=${r}`);
    }
    // Newtonian limit; the zero-torque inner boundary leaves an O(r^-1/2) relative correction
    const r = 1e9;
    close(pageThorneFlux(a, r) * 8 * Math.PI * r ** 3 / (3 * (1 - Math.sqrt(isco(a) / r))), 1, 1e-4, "Newtonian");
    // luminosity at infinity
    let Lum = 0;
    const rms = isco(a);
    const panels: [number, number][] = [];
    for (let i = 0; i < 200; i++) panels.push([rms + (i / 200) ** 2 * 100, rms + ((i + 1) / 200) ** 2 * 100]);
    for (let i = 0; i < 200; i++) panels.push([(rms + 100) * 1e6 ** (i / 200), (rms + 100) * 1e6 ** ((i + 1) / 200)]);
    const gl = [[-0.7745966692414834, 5 / 9], [0, 8 / 9], [0.7745966692414834, 5 / 9]];
    for (const [r0, r1] of panels)
      for (const [xg, wg] of gl) {
        const rr = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * xg;
        Lum += wg * 0.5 * (r1 - r0) * 4 * Math.PI * rr * circularOrbit(a, rr).E * pageThorneFlux(a, rr);
      }
    Lum += 1.5 / ((rms + 100) * 1e6); // Newtonian tail
    close(Lum, efficiency(a), 2e-5 * efficiency(a), `luminosity a=${a}`);
  }
  // 10 Msun at 10% Eddington: Newtonian Shakura–Sunyaev peak 0.488 (3GM Mdot / 8 pi sigma r_in^3)^1/4 = 4.86e6 K;
  // relativistic Novikov–Thorne is ~20% cooler. T scales as (lambda / M)^(1/4).
  const Tpk = (M: number, l: number) => Math.max(...Array.from({ length: 400 }, (_, i) => diskTemperature(0, 6 + i * 0.05, M, l)));
  const T10 = Tpk(10, 0.1);
  assert.ok(T10 > 0.75 * 4.86e6 && T10 < 0.85 * 4.86e6, `T_peak ${T10}`);
  close(Tpk(1e9, 1e-3) / T10, (1e-8 * 1e-2) ** 0.25, 1e-9);
});

test("radial free fall from rest at r0 = 10M crosses the horizon on the cycloid tau(r)", () => {
  const ob = new Observer({ a: 0, mode: "static", r: 10, theta: Math.PI / 2, prograde: true, gyro: false });
  ob.release();
  const eta = Math.acos(2 * 1.0 / 10 - 1);
  const tau = Math.sqrt(1000 / 8) * (eta + Math.sin(eta));
  for (let i = 0; i < 100; i++) ob.advance(tau / 100);
  close(ob.rNow, 1.0, 1e-7);
});

test("Kerr free fall conserves E, L, Q and normalisation; gyroscope geodetic precession", () => {
  const a = 0.9;
  const ob = new Observer({ a, mode: "zamo", r: 8, theta: 1.0, prograde: true, gyro: true });
  ob.release();
  const st = () => constants(a, [...ob.pos, ...lower(metric(a, ob.pos), ob.u)]);
  const c0 = st();
  for (let i = 0; i < 30; i++) ob.advance(1);
  const c1 = st();
  close(c1.E, c0.E, 1e-9, "E");
  close(c1.L, c0.L, 1e-9, "L");
  close(c1.Q, c0.Q, 1e-7, "Q");
  close(c1.mu2, 1, 1e-9, "u.u");

  // Schwarzschild circular orbit at r = 10: spin precesses by 2 pi (1 - sqrt(1 - 3M/r)) per orbit
  const r = 10;
  const orb = new Observer({ a: 0, mode: "orbit", r, theta: Math.PI / 2, prograde: true, gyro: true });
  const e0 = orb.referenceTriad()[0];
  orb.release();
  const period = (2 * Math.PI / circularOrbit(0, r).omega) / circularOrbit(0, r).ut;
  for (let i = 0; i < 200; i++) orb.advance(period / 200);
  const g = metric(0, orb.pos);
  close(Math.hypot(orb.pos[1] - 10, orb.pos[2]), 0, 1e-6, "back at start");
  const e1 = triad(g, orb.u, orb.gyroTriad!)[0];
  close(Math.acos(dot(g, e0, e1)), 2 * Math.PI * (1 - Math.sqrt(1 - 3 / r)), 1e-6, "geodetic precession");
});

test("tidal tensor: Schwarzschild eigenvalues, vacuum trace, Kretschmann invariant", () => {
  const ob = new Observer({ a: 0, mode: "static", r: 10, theta: 1.0, prograde: true, gyro: false });
  const ev = eigSym3(tidalTensors(0, ob.pos, ob.camera().frame).E);
  close(ev[0], -2e-3, 1e-9);
  close(ev[1], 1e-3, 1e-9);
  close(ev[2], 1e-3, 1e-9);
  const a = 0.9;
  const ob2 = new Observer({ a, mode: "zamo", r: 3, theta: 0.8, prograde: true, gyro: false });
  ob2.release();
  ob2.thrust = [0, 0.05, 0];
  ob2.advance(2); // an accelerated, boosted observer
  const { E, B } = tidalTensors(a, ob2.pos, ob2.camera().frame);
  close(E[0][0] + E[1][1] + E[2][2], 0, 1e-8, "trace");
  let ee = 0, bb2 = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { ee += E[i][j] ** 2; bb2 += B[i][j] ** 2; }
  const rr = ob2.rNow, c = ob2.pos[3] / rr;
  const K = 48 * (rr ** 6 - 15 * a * a * rr ** 4 * c * c + 15 * a ** 4 * rr * rr * c ** 4 - a ** 6 * c ** 6) / (rr * rr + a * a * c * c) ** 6;
  close(8 * (ee - bb2), K, 1e-6 * Math.abs(K), "Kretschmann");
});

test("blackbody chromaticity: CIE illuminant A is Planckian at 2856 K", () => {
  const [X, Y, Z] = blackbodyXYZ(2856);
  close(X / (X + Y + Z), 0.44757, 2e-4);
  close(Y / (X + Y + Z), 0.40745, 2e-4);
});

test("camera stays right-handed (right = forward x up) under look rotations", () => {
  const ob = new Observer({ a: 0.5, mode: "static", r: 1e5, theta: 1.1, prograde: true, gyro: false });
  for (const [yaw, pitch] of [[0, 0], [0.3, -0.2], [2.5, 1.0], [-1.2, 0.4]]) {
    ob.rotate(yaw, pitch);
    const [, f, r, u] = ob.camera().frame;
    const c = [f[2] * u[3] - f[3] * u[2], f[3] * u[1] - f[1] * u[3], f[1] * u[2] - f[2] * u[1]];
    for (let i = 0; i < 3; i++) close(c[i], r[i + 1], 1e-4, `yaw ${yaw}`); // flat to O(M/r)
  }
});
