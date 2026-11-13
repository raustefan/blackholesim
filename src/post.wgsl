// Post-processing: auto-exposure (luminance histogram), bloom (camera PSF stand-in), AgX display transform,
// photographic finish, and the Luminet (1979) stipple plate.

// ------------------------------------------------------------------ exposure

struct ExpoParams { size: vec2u, dt: f32, evComp: f32, minLog: f32, maxLog: f32, rate: f32, _pad: f32 };

@group(0) @binding(0) var hdrIn: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>, 256>;
@group(0) @binding(2) var<storage, read_write> expoBuf: array<f32>; // [0] pre-exposure, [1] mean log2 L, [2] init
@group(0) @binding(3) var<uniform> EP: ExpoParams;

var<workgroup> localHist: array<atomic<u32>, 256>;
var<workgroup> counts: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  atomicStore(&localHist[li], 0u);
  workgroupBarrier();
  if (all(gid.xy < EP.size)) {
    let c = textureLoad(hdrIn, gid.xy, 0).rgb;
    let Y = dot(c, vec3f(0.2126, 0.7152, 0.0722)) / expoBuf[0]; // absolute luminance
    var bin = 0u;
    if (Y > 0.0) {
      bin = u32(clamp((log2(Y) - EP.minLog) / (EP.maxLog - EP.minLog) * 254.0 + 1.0, 1.0, 255.0));
    }
    atomicAdd(&localHist[bin], 1u);
  }
  workgroupBarrier();
  atomicAdd(&hist[li], atomicLoad(&localHist[li]));
}

/** Meter: mean log luminance of the 40th..98th percentile of lit pixels, highlight-protected; exponential adaptation. */
@compute @workgroup_size(256)
fn adapt(@builtin(local_invocation_index) li: u32) {
  counts[li] = select(atomicLoad(&hist[li]), 0u, li == 0u);
  atomicStore(&hist[li], 0u);
  workgroupBarrier();
  if (li == 0u) {
    var total = 0.0;
    for (var i = 1u; i < 256u; i++) { total += f32(counts[i]); }
    let lo = 0.4 * total;
    let hi = 0.98 * total;
    var acc = 0.0;
    var sum = 0.0;
    var wsum = 0.0;
    var p995 = EP.minLog;
    for (var i = 1u; i < 256u; i++) {
      let c = f32(counts[i]);
      let L = EP.minLog + (f32(i) - 0.5) / 254.0 * (EP.maxLog - EP.minLog);
      let w = max(0.0, min(acc + c, hi) - max(acc, lo));
      if (acc < 0.995 * total && acc + c >= 0.995 * total) { p995 = L; }
      acc += c;
      sum += w * L;
      wsum += w;
    }
    if (wsum > 0.0) {
      // highlight protection: a small bright disk in a dark sky must not be blown out
      let goal = max(sum / wsum, p995 - 2.5);
      let k = 1.0 - exp(-EP.dt * EP.rate);
      expoBuf[1] = select(goal, expoBuf[1] + (goal - expoBuf[1]) * k, expoBuf[2] > 0.0);
      expoBuf[2] = 1.0;
    }
    expoBuf[0] = 0.18 * exp2(EP.evComp - expoBuf[1]);
  }
}

// ------------------------------------------------------------------ fullscreen helpers

struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vsFull(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return VO(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}

// ------------------------------------------------------------------ bloom (Jimenez 2014 dual filter)

struct BloomP { texel: vec2f, uvScale: vec2f, karis: f32, radius: f32, _p: vec2f };

@group(0) @binding(0) var bSrc: texture_2d<f32>;
@group(0) @binding(1) var bLin: sampler;
@group(0) @binding(2) var<uniform> BP: BloomP;

fn tap(uv: vec2f) -> vec3f {
  return textureSampleLevel(bSrc, bLin, min(uv, BP.uvScale - 0.5 * BP.texel), 0.0).rgb;
}

fn karisW(c: vec3f) -> f32 { return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722))); }

@fragment
fn bloomDown(in: VO) -> @location(0) vec4f {
  let uv = in.uv * BP.uvScale;
  let t = BP.texel;
  let a = tap(uv + t * vec2f(-2, -2)); let b = tap(uv + t * vec2f(0, -2)); let c = tap(uv + t * vec2f(2, -2));
  let d = tap(uv + t * vec2f(-2, 0));  let e = tap(uv);                     let f = tap(uv + t * vec2f(2, 0));
  let g = tap(uv + t * vec2f(-2, 2));  let h = tap(uv + t * vec2f(0, 2));  let i = tap(uv + t * vec2f(2, 2));
  let j = tap(uv + t * vec2f(-1, -1)); let k = tap(uv + t * vec2f(1, -1));
  let l = tap(uv + t * vec2f(-1, 1));  let m = tap(uv + t * vec2f(1, 1));
  if (BP.karis > 0.0) { // firefly suppression on the first level
    let g0 = (j + k + l + m) * 0.25; let g1 = (a + b + d + e) * 0.25; let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25; let g4 = (e + f + h + i) * 0.25;
    let w0 = karisW(g0) * 0.5; let w1 = karisW(g1) * 0.125; let w2 = karisW(g2) * 0.125;
    let w3 = karisW(g3) * 0.125; let w4 = karisW(g4) * 0.125;
    return vec4f((g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4), 1.0);
  }
  return vec4f(e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125, 1.0);
}

@fragment
fn bloomUp(in: VO) -> @location(0) vec4f {
  let uv = in.uv;
  let t = BP.texel * BP.radius;
  var s = tap(uv) * 4.0;
  s += (tap(uv + t * vec2f(-1, 0)) + tap(uv + t * vec2f(1, 0)) + tap(uv + t * vec2f(0, -1)) + tap(uv + t * vec2f(0, 1))) * 2.0;
  s += tap(uv + t * vec2f(-1, -1)) + tap(uv + t * vec2f(1, -1)) + tap(uv + t * vec2f(-1, 1)) + tap(uv + t * vec2f(1, 1));
  return vec4f(s / 16.0, 1.0);
}

// ------------------------------------------------------------------ composite

struct CompP { uvScale: vec2f, res: vec2f, bloom: f32, time: f32, grain: f32, vignette: f32, luminet: u32, levels: f32, _p: vec2f };

@group(0) @binding(0) var cHdr: texture_2d<f32>;
@group(0) @binding(1) var cBloom: texture_2d<f32>;
@group(0) @binding(2) var cLin: sampler;
@group(0) @binding(3) var<uniform> CP: CompP;

// AgX (Sobotka), polynomial fit by B. Wrensch; output is display-encoded (~sRGB).
fn agx(val: vec3f) -> vec3f {
  let inset = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                      0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let outset = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var v = inset * max(val, vec3f(0.0));
  v = clamp(log2(max(v, vec3f(1e-10))), vec3f(minEv), vec3f(maxEv));
  v = (v - minEv) / (maxEv - minEv);
  let v2 = v * v;
  let v4 = v2 * v2;
  v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
  // "punchy" look (power 1.35, saturation 1.4), as in Blender's AgX looks
  v = pow(max(v, vec3f(0.0)), vec3f(1.35));
  let luma = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  v = luma + 1.4 * (v - luma);
  return clamp(outset * v, vec3f(0.0), vec3f(1.0));
}

fn hash2(p: vec2f) -> f32 {
  var q = fract(vec3f(p.xyx) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

@fragment
fn composite(in: VO) -> @location(0) vec4f {
  let uv = in.pos.xy / CP.res;
  var c = textureSampleLevel(cHdr, cLin, uv * CP.uvScale, 0.0).rgb;
  let b = textureSampleLevel(cBloom, cLin, uv, 0.0).rgb / CP.levels;
  c = mix(c, b, CP.bloom);
  var o = agx(c);
  if (CP.luminet != 0u) {
    // Luminet's plate: ink dots whose density follows the bolometric intensity
    let cell = 2.6;
    let id = floor(in.pos.xy / cell);
    let v = dot(o, vec3f(0.2126, 0.7152, 0.0722));
    var ink = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = id + vec2f(f32(dx), f32(dy));
        let ctr = (q + 0.5 + 0.35 * (vec2f(hash2(q), hash2(q + 17.0)) - 0.5)) * cell;
        let on = step(hash2(q + 3.1), pow(v, 0.8));
        ink = max(ink, on * (1.0 - smoothstep(0.55, 1.05, length(in.pos.xy - ctr))));
      }
    }
    o = mix(vec3f(0.035, 0.03, 0.028), vec3f(0.93, 0.9, 0.82), ink);
  }
  // photographic finish: vignette and luminance-dependent grain
  let d = (uv - 0.5) * vec2f(CP.res.x / CP.res.y, 1.0);
  o *= 1.0 - CP.vignette * smoothstep(0.35, 1.1, length(d));
  let n = hash2(in.pos.xy + fract(CP.time * 7.13) * 917.0) - 0.5;
  o += CP.grain * n * (0.35 + 0.65 * (1.0 - abs(2.0 * dot(o, vec3f(0.333)) - 1.0)));
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}
