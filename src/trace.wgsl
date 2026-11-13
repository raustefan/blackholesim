// Kerr ray tracer. Single-precision port of src/kerr.ts (traceRay, escapeDirection), verified against it by
// test/gpu.test.ts. Every pixel integrates the exact null geodesic backwards from the camera in ingoing
// Kerr–Schild coordinates with an adaptive Dormand–Prince 5(4) integrator. Units G = c = M = 1.

struct Params {
  pos: vec4f,                          // camera event (t, x, y, z)
  e0: vec4f, e1: vec4f, e2: vec4f, e3: vec4f, // camera tetrad: u, forward, right, up (contravariant)
  sky0: vec4f, sky1: vec4f, sky2: vec4f, // columns of the rotation hole frame -> equatorial sky frame
  res: vec2f, jitter: vec2f,
  tanHalf: f32, aspect: f32, a: f32, bend: f32,
  diskIn: f32, diskOut: f32, rEscape: f32, tol: f32,
  maxSteps: u32, flags: u32, skyGain: f32, turbPeriod: f32,
  cellN: u32, cellMin: f32, inside: u32, _pad: f32,
};

const F_DOPPLER = 1u;    // longitudinal Doppler shift / beaming of disk light
const F_REDSHIFT = 2u;   // all frequency shifts (off: g = 1)
const F_LUMINET = 4u;    // bolometric disk intensity only (Luminet 1979)
const F_TURB = 8u;       // artistic turbulence texture on the disk
const F_STARS = 16u;
const F_DISK = 32u;
const F_DEBUG = 64u;     // raw ray results for the GPU test

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> bbLut: array<vec4f>;   // ln X, ln Y, ln Z of a blackbody vs log10 T
@group(0) @binding(2) var<storage, read> diskT: array<f32>;     // T_eff [K] at log-spaced r in [diskIn, diskOut]
@group(0) @binding(3) var skyDiffuse: texture_cube<f32>;        // linear sRGB cd/m^2 * 1e4, equatorial frame
@group(0) @binding(4) var skyCoarse: texture_cube<f32>;         // catalogue starlight XYZ cd/m^2 * 1e4
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var<storage, read> stars: array<vec4f>;   // unit direction (equatorial), A = flux / Y_bb(T)
@group(0) @binding(7) var<storage, read> starT: array<f32>;
@group(0) @binding(8) var<storage, read> cells: array<u32>;     // star index offsets per cube cell (+1)
@group(0) @binding(9) var<storage, read> expo: array<f32>;      // [0] pre-exposure applied to the output
@group(0) @binding(10) var<storage, read> chromaT: array<f32>;  // [x0, x1, T(x) * 256]: Planckian T from CIE x

const LUT_T0 = 2.5;
const LUT_T1 = 9.5;
const LUT_N = 1024u;
const DISK_N = 512u;
const SKY_SCALE = 1e-4;
const PI = 3.14159265358979;
const SIGMA_SB = 5.670374419e-8;

const XYZ2RGB = mat3x3f(3.2404542, -0.969266, 0.0556434, -1.5371385, 1.8760108, -0.2040259, -0.4985314, 0.041556, 1.0572252);
const RGB2XYZ = mat3x3f(0.4124564, 0.2126729, 0.0193339, 0.3575761, 0.7151522, 0.119192, 0.1804375, 0.072175, 0.9503041);

// ------------------------------------------------------------------ colour

fn bbXYZ(T: f32) -> vec3f {
  let lt = log2(max(T, 1.0)) * 0.30102999566;
  if (lt < LUT_T0) { return vec3f(0.0); }
  let u = clamp((lt - LUT_T0) / (LUT_T1 - LUT_T0), 0.0, 1.0) * f32(LUT_N - 1u);
  let i = min(u32(u), LUT_N - 2u);
  return exp(mix(bbLut[i].xyz, bbLut[i + 1u].xyz, u - f32(i)));
}

/** Frequency shift g applied to light whose spectrum is approximated by a Planckian of the same chromaticity. */
fn shiftXYZ(c: vec3f, g: f32) -> vec3f {
  let s = c.x + c.y + c.z;
  if (abs(g - 1.0) < 1e-5 || s <= 0.0) { return c; }
  let u = clamp((c.x / s - chromaT[0]) / (chromaT[1] - chromaT[0]), 0.0, 1.0) * 255.0;
  let i = min(u32(u), 254u);
  let T = mix(chromaT[2u + i], chromaT[3u + i], u - f32(i));
  return c * bbXYZ(g * T) / bbXYZ(T);
}

// ------------------------------------------------------------------ Kerr–Schild geometry

fn rOf(p: vec3f) -> f32 {
  let a2 = P.a * P.a;
  let w = dot(p, p) - a2;
  let s = sqrt(w * w + 4.0 * a2 * p.z * p.z);
  let r2 = select(2.0 * a2 * p.z * p.z / max(s - w, 1e-30), 0.5 * (w + s), w >= 0.0);
  return sqrt(r2);
}

fn ksL(p: vec3f, r: f32) -> vec3f {
  let D = r * r + P.a * P.a;
  return vec3f((r * p.x + P.a * p.y) / D, (r * p.y - P.a * p.x) / D, p.z / r);
}

fn ksF(p: vec3f, r: f32) -> f32 {
  let r2 = r * r;
  return P.bend * 2.0 * r2 * r / (r2 * r2 + P.a * P.a * p.z * p.z);
}

struct D { x: vec4f, k: vec3f };

/** Backward-in-affine-parameter Hamiltonian flow for covariant momentum (kt, k). */
fn rhs(x: vec4f, k: vec3f, kt: f32) -> D {
  let p = x.yzw;
  let a = P.a;
  let a2 = a * a;
  let w = dot(p, p) - a2;
  let s = sqrt(w * w + 4.0 * a2 * p.z * p.z);
  let r2 = select(2.0 * a2 * p.z * p.z / max(s - w, 1e-30), 0.5 * (w + s), w >= 0.0);
  let r = sqrt(r2);
  let Dn = r2 + a2;
  let dr = vec3f(r2 * p.x, r2 * p.y, (r2 + a2) * p.z) / (r * s);
  let den = r2 * r2 + a2 * p.z * p.z;
  let f = P.bend * 2.0 * r2 * r / den;
  let df = f * (3.0 * dr / r - (4.0 * r2 * r * dr + vec3f(0.0, 0.0, 2.0 * a2 * p.z)) / den);
  let l = vec3f((r * p.x + a * p.y) / Dn, (r * p.y - a * p.x) / Dn, p.z / r);
  let lp = -kt + dot(l, k);
  let C = (k.x * p.x + k.y * p.y) / Dn - 2.0 * r * (k.x * l.x + k.y * l.y) / Dn - k.z * p.z / r2;
  let dlp = dr * C + vec3f(r * k.x - a * k.y, r * k.y + a * k.x, 0.0) / Dn + vec3f(0.0, 0.0, k.z / r);
  let dx = vec4f(-kt + f * lp, k - f * lp * l);
  let dk = 0.5 * lp * lp * df + f * lp * dlp;
  return D(-dx, -dk);
}

struct S { x: vec4f, k: vec3f, err: f32, d7: D };

// Dormand–Prince 5(4); FSAL: d1 is the derivative at (x, k), d7 the derivative at the result.
fn dpStep(x: vec4f, k: vec3f, kt: f32, h: f32, d1: D) -> S {
  let k2 = rhs(x + h * (0.2 * d1.x), k + h * (0.2 * d1.k), kt);
  let k3 = rhs(x + h * (3.0 / 40.0 * d1.x + 9.0 / 40.0 * k2.x), k + h * (3.0 / 40.0 * d1.k + 9.0 / 40.0 * k2.k), kt);
  let k4 = rhs(x + h * (44.0 / 45.0 * d1.x - 56.0 / 15.0 * k2.x + 32.0 / 9.0 * k3.x),
               k + h * (44.0 / 45.0 * d1.k - 56.0 / 15.0 * k2.k + 32.0 / 9.0 * k3.k), kt);
  let k5 = rhs(x + h * (19372.0 / 6561.0 * d1.x - 25360.0 / 2187.0 * k2.x + 64448.0 / 6561.0 * k3.x - 212.0 / 729.0 * k4.x),
               k + h * (19372.0 / 6561.0 * d1.k - 25360.0 / 2187.0 * k2.k + 64448.0 / 6561.0 * k3.k - 212.0 / 729.0 * k4.k), kt);
  let k6 = rhs(x + h * (9017.0 / 3168.0 * d1.x - 355.0 / 33.0 * k2.x + 46732.0 / 5247.0 * k3.x + 49.0 / 176.0 * k4.x - 5103.0 / 18656.0 * k5.x),
               k + h * (9017.0 / 3168.0 * d1.k - 355.0 / 33.0 * k2.k + 46732.0 / 5247.0 * k3.k + 49.0 / 176.0 * k4.k - 5103.0 / 18656.0 * k5.k), kt);
  let x5 = x + h * (35.0 / 384.0 * d1.x + 500.0 / 1113.0 * k3.x + 125.0 / 192.0 * k4.x - 2187.0 / 6784.0 * k5.x + 11.0 / 84.0 * k6.x);
  let q5 = k + h * (35.0 / 384.0 * d1.k + 500.0 / 1113.0 * k3.k + 125.0 / 192.0 * k4.k - 2187.0 / 6784.0 * k5.k + 11.0 / 84.0 * k6.k);
  let k7 = rhs(x5, q5, kt);
  let e = h * (71.0 / 57600.0 * d1.x - 71.0 / 16695.0 * k3.x + 71.0 / 1920.0 * k4.x - 17253.0 / 339200.0 * k5.x + 22.0 / 525.0 * k6.x - 1.0 / 40.0 * k7.x);
  let err = max(abs(e.y), max(abs(e.z), abs(e.w))) / (1e-3 + rOf(x.yzw));
  return S(x5, q5, err, k7);
}

/** u^t of a circular equatorial emitter with angular velocity om (from the metric; bend-aware). */
fn diskUt(p: vec3f, om: f32) -> f32 {
  let r = rOf(p);
  let l = ksL(p, r);
  let lv = 1.0 + om * (p.x * l.y - p.y * l.x);
  return inverseSqrt(max(1.0 - om * om * (p.x * p.x + p.y * p.y) - ksF(p, r) * lv * lv, 1e-12));
}

// remaining Schwarzschild bending to infinity (see kerr.ts escapeDirection); pk along travel, E > 0
fn escapeDir(x: vec3f, pk: vec3f, E: f32) -> vec3f {
  let r = length(x);
  let xh = x / r;
  let Lv = cross(x, pk);
  let Ln = length(Lv);
  if (Ln < 1e-7 * r * E) { return xh; }
  let b = Ln / E;
  let t = cross(Lv, xh) / Ln;
  let chi0 = asin(min(1.0, b / r));
  var dphi = 0.0;
  var gx = array<f32, 5>(0.0, -0.5384693101056831, 0.5384693101056831, -0.906179845938664, 0.906179845938664);
  var gw = array<f32, 5>(0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850560891, 0.2369268850560891);
  for (var k = 0; k < 8; k++) {
    for (var j = 0; j < 5; j++) {
      let c = chi0 * (f32(k) + 0.5 + 0.5 * gx[j]) / 8.0;
      let sn = sin(c);
      let co = cos(c);
      dphi += gw[j] * 0.5 * chi0 / 8.0 * co * inverseSqrt(co * co + 2.0 * sn * sn * sn / b);
    }
  }
  return cos(dphi) * xh + sin(dphi) * t;
}

// ------------------------------------------------------------------ disk

fn hash3(p: vec3f) -> f32 {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3f(1, 0, 0)), u.x), mix(hash3(i + vec3f(0, 1, 0)), hash3(i + vec3f(1, 1, 0)), u.x), u.y),
             mix(mix(hash3(i + vec3f(0, 0, 1)), hash3(i + vec3f(1, 0, 1)), u.x), mix(hash3(i + vec3f(0, 1, 1)), hash3(i + vec3f(1, 1, 1)), u.x), u.y), u.z);
}

fn fbm(p: vec3f) -> f32 {
  var s = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < 5; i++) {
    s += amp * vnoise(q);
    q = q * 2.03 + vec3f(1.7, 9.2, 3.1);
    amp *= 0.5;
  }
  return s;
}

/** Artistic (non-physical) turbulence: a flux multiplier with mean ~1, advected at the local orbital speed. */
fn turbulence(r: f32, phi: f32, t: f32, om: f32) -> f32 {
  let tt = t / P.turbPeriod;
  var m = 0.0;
  for (var layer = 0; layer < 2; layer++) {
    let f = fract(tt + 0.5 * f32(layer));
    let w = 1.0 - abs(2.0 * f - 1.0);
    let ph = phi - om * f * P.turbPeriod;
    let seed = floor(tt + 0.5 * f32(layer)) * 17.0 + f32(layer) * 5.0;
    let q = vec3f(cos(ph) * 2.5, sin(ph) * 2.5, log(r) * 9.0 + seed);
    m += w * fbm(q);
  }
  return clamp(0.35 + 1.6 * m * m * 2.0, 0.05, 4.0);
}

fn diskTemp(r: f32) -> f32 {
  let u = clamp(log(r / P.diskIn) / log(P.diskOut / P.diskIn), 0.0, 1.0) * f32(DISK_N - 1u);
  let i = min(u32(u), DISK_N - 2u);
  return mix(diskT[i], diskT[i + 1u], u - f32(i));
}

/**
 * Cube-map lookup with the mip level chosen from the ray differentials (sx, sy). Explicit LOD rather than
 * gradient sampling: Firefox's naga emits invalid Metal for gradient sampling of cube textures.
 */
fn sampleCube(t: texture_cube<f32>, d: vec3f, sx: vec3f, sy: vec3f) -> vec3f {
  let texel = 2.0 / f32(textureDimensions(t).x); // angular texel size at a face centre
  let lod = log2(max(max(length(sx), length(sy)) / texel, 1e-8));
  return textureSampleLevel(t, samp, d, lod).rgb;
}

// ------------------------------------------------------------------ point stars

fn cellOf(d: vec3f) -> u32 {
  let ad = abs(d);
  var face = 0u;
  var sc = 0.0;
  var tc = 0.0;
  var ma = 0.0;
  if (ad.x >= ad.y && ad.x >= ad.z) {
    ma = ad.x;
    if (d.x > 0.0) { face = 0u; sc = -d.z; tc = -d.y; } else { face = 1u; sc = d.z; tc = -d.y; }
  } else if (ad.y >= ad.z) {
    ma = ad.y;
    if (d.y > 0.0) { face = 2u; sc = d.x; tc = d.z; } else { face = 3u; sc = d.x; tc = -d.z; }
  } else {
    ma = ad.z;
    if (d.z > 0.0) { face = 4u; sc = d.x; tc = -d.y; } else { face = 5u; sc = -d.x; tc = -d.y; }
  }
  let n = f32(P.cellN);
  let u = u32(clamp((sc / ma * 0.5 + 0.5) * n, 0.0, n - 1.0));
  let v = u32(clamp((tc / ma * 0.5 + 0.5) * n, 0.0, n - 1.0));
  return face * P.cellN * P.cellN + v * P.cellN + u;
}

/**
 * Catalogue stars as true point sources. A star of flux F lands on the pixel grid through the local
 * sky<-pixel Jacobian (sx, sy); pixel radiance = F K(delta) / Omega_sky_pix, so lensing magnification
 * (Omega_cam / Omega_sky) brightens stars without blurring them.
 */
fn pointStars(s: vec3f, sx: vec3f, sy: vec3f, rk: f32, g: f32) -> vec3f {
  let a11 = dot(sx, sx);
  let a12 = dot(sx, sy);
  let a22 = dot(sy, sy);
  let det = a11 * a22 - a12 * a12;
  let omegaPix = length(cross(sx, sy));
  if (det <= 1e-6 * a11 * a22 || omegaPix <= 0.0) { return vec3f(0.0); }
  let t1 = normalize(cross(s, select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(s.z) > 0.9)));
  let t2 = cross(s, t1);
  var seen = array<u32, 9>();
  var nseen = 0u;
  var sum = vec3f(0.0);
  let sig2 = 0.5 * 0.5;
  for (var i = -1; i <= 1; i++) {
    for (var j = -1; j <= 1; j++) {
      let c = cellOf(normalize(s + rk * (f32(i) * t1 + f32(j) * t2)));
      var dup = false;
      for (var m = 0u; m < nseen; m++) { dup = dup || seen[m] == c; }
      if (dup) { continue; }
      seen[nseen] = c;
      nseen++;
      for (var m = cells[c]; m < cells[c + 1u]; m++) {
        let st = stars[m];
        let dv = st.xyz - s;
        let bx = dot(sx, dv);
        let by = dot(sy, dv);
        let px = (a22 * bx - a12 * by) / det;
        let py = (a11 * by - a12 * bx) / det;
        let rr = px * px + py * py;
        if (rr < 9.0 * sig2 * 2.0) {
          sum += st.w * bbXYZ(g * starT[m]) * exp(-rr / (2.0 * sig2)) / (2.0 * PI * sig2);
        }
      }
    }
  }
  return sum / omegaPix;
}

// ------------------------------------------------------------------ tracing

const HOLE = 0u;
const ESCAPE = 1u;
const DISK = 2u;

struct Hit { kind: u32, dir: vec3f, g: f32, xyz: vec3f, r: f32 };

fn trace(px: vec2f) -> Hit {
  var hit = Hit(HOLE, vec3f(0.0, 0.0, 1.0), 1.0, vec3f(0.0), 0.0);
  let ndc = vec2f(2.0 * px.x / P.res.x - 1.0, 1.0 - 2.0 * px.y / P.res.y);
  let n = normalize(vec3f(1.0, ndc.x * P.tanHalf * P.aspect, ndc.y * P.tanHalf));
  // photon arriving from direction n: k = u - n^i e_i, lowered with the metric at the camera
  let kv = P.e0 - n.x * P.e1 - n.y * P.e2 - n.z * P.e3;
  let p0 = P.pos.yzw;
  let r0 = rOf(p0);
  let l0 = ksL(p0, r0);
  let lk = kv.x + dot(l0, kv.yzw);
  let f0 = ksF(p0, r0);
  let kt = -kv.x + f0 * lk;
  var k = kv.yzw + f0 * lk * l0;
  var x = P.pos;
  let rp = 1.0 + sqrt(max(0.0, 1.0 - P.a * P.a));
  var h = 0.05 * max(1.0, r0);
  var d1 = rhs(x, k, kt);
  let diskOn = (P.flags & F_DISK) != 0u;
  for (var i = 0u; i < P.maxSteps; i++) {
    let st = dpStep(x, k, kt, h, d1);
    let en = st.err / P.tol;
    if (en > 1.0 && h > 1e-6) {
      h *= max(0.2, 0.9 * pow(en, -0.2));
      continue;
    }
    let px0 = x;
    let pk0 = k;
    let pd0 = d1;
    let hTaken = h;
    x = st.x;
    k = st.k;
    d1 = st.d7;
    h *= select(min(5.0, max(0.2, 0.9 * pow(en, -0.2))), 5.0, en == 0.0);
    let r = rOf(x.yzw);
    h = min(h, 0.5 * r + 0.5);

    if (diskOn && px0.w * x.w <= 0.0 && px0.w != x.w) {
      // Illinois regula falsi on the step fraction for z = 0
      var s0 = 0.0;
      var z0 = px0.w;
      var s1 = 1.0;
      var z1 = x.w;
      var xc = x;
      var kc = k;
      var side = 0;
      for (var it = 0; it < 5; it++) {
        let sf = (s0 * z1 - s1 * z0) / (z1 - z0);
        let sr = dpStep(px0, pk0, kt, sf * hTaken, pd0);
        xc = sr.x;
        kc = sr.k;
        if (xc.w * z1 > 0.0) {
          s1 = sf; z1 = xc.w;
          if (side == -1) { z0 *= 0.5; }
          side = -1;
        } else {
          s0 = sf; z0 = xc.w;
          if (side == 1) { z1 *= 0.5; }
          side = 1;
        }
      }
      let rc = rOf(xc.yzw);
      if (rc >= P.diskIn && rc <= P.diskOut) {
        let om = 1.0 / (pow(rc, 1.5) + P.a);
        let ut = diskUt(xc.yzw, om);
        let kphi = xc.y * kc.y - xc.z * kc.x;
        let Einf = -kt;
        var g = 1.0;
        if ((P.flags & F_REDSHIFT) != 0u) {
          g = 1.0 / (Einf * ut);
          if ((P.flags & F_DOPPLER) != 0u) { g /= 1.0 - om * kphi / Einf; }
        }
        var T = diskTemp(rc);
        if ((P.flags & F_TURB) != 0u) {
          let phi = atan2(xc.z, xc.y) - atan2(P.a, rc); // Kerr–Schild azimuth
          T *= pow(turbulence(rc, phi, xc.x, om), 0.25);
        }
        hit.kind = DISK;
        hit.g = g;
        hit.r = rc;
        if ((P.flags & F_LUMINET) != 0u) {
          let Tb = g * T; // bolometric observed intensity g^4 sigma T^4 / pi, shown as grey
          hit.xyz = vec3f(SIGMA_SB * Tb * Tb * Tb * Tb / PI) * vec3f(0.9505, 1.0, 1.089);
        } else {
          hit.xyz = bbXYZ(g * T);
        }
        return hit;
      }
    }
    if (r > P.rEscape) {
      if (dot(x.yzw, x.yzw - px0.yzw) > 0.0) {
        hit.kind = ESCAPE;
        hit.dir = escapeDir(x.yzw, -k, -kt);
        hit.g = select(1.0, 1.0 / -kt, (P.flags & F_REDSHIFT) != 0u);
        return hit;
      }
    }
    if (length(k) > 1e4 * abs(kt) || (P.bend == 0.0 && r < rp) || (P.inside == 0u && r < 0.98 * rp)) {
      return hit; // event horizon: infinitely redshifted
    }
  }
  return hit;
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let hit = trace(fc.xy + P.jitter);
  // ray differentials from the neighbouring pixels of the quad (uniform control flow)
  let ddx = dpdxFine(hit.dir);
  let ddy = dpdyFine(hit.dir);
  if ((P.flags & F_DEBUG) != 0u) {
    if (hit.kind == ESCAPE) { return vec4f(hit.dir, hit.g); }
    if (hit.kind == DISK) { return vec4f(hit.r, hit.g, 0.0, -1.0); }
    return vec4f(0.0);
  }
  var xyz = hit.xyz;
  if (hit.kind == ESCAPE && (P.flags & F_LUMINET) == 0u) {
    let M = mat3x3f(P.sky0.xyz, P.sky1.xyz, P.sky2.xyz);
    let s = M * hit.dir;
    let sx = M * ddx;
    let sy = M * ddy;
    var diffuse = RGB2XYZ * sampleCube(skyDiffuse, s, sx, sy);
    var pts = vec3f(0.0);
    if ((P.flags & F_STARS) != 0u) {
      let rk = 1.5 * sqrt(dot(sx, sx) + dot(sy, sy));
      let w = smoothstep(0.5 * P.cellMin, P.cellMin, rk);
      if (w < 1.0) { pts = (1.0 - w) * pointStars(s, sx, sy, rk, hit.g); }
      if (w > 0.0) { diffuse += w * sampleCube(skyCoarse, s, sx, sy); }
    }
    xyz = P.skyGain * (shiftXYZ(diffuse * SKY_SCALE, hit.g) + pts);
  }
  let rgb = XYZ2RGB * xyz * expo[0];
  return vec4f(min(rgb, vec3f(6e4)), 1.0);
}
