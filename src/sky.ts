// The celestial sphere: diffuse Milky Way cube (NASA SVS Deep Star Maps, catalogue stars removed) and the HYG
// catalogue as point sources. See scripts/prepare_data.py. Everything here is in the equatorial frame.
import { LUX_MAG0, blackbodyXYZ, bvToTemperature } from "./color.ts";

export const STAR_CELLS = 128; // per cube face
export const STORE_SCALE = 1e4; // sky textures hold cd/m^2 * 1e4
export const COARSE_SIZE = 256;

/** Same cell index as trace.wgsl cellOf (WebGPU cube-face conventions). */
export function cellOf(x: number, y: number, z: number, n = STAR_CELLS): number {
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  let face, sc, tc, ma;
  if (ax >= ay && ax >= az) { ma = ax; [face, sc, tc] = x > 0 ? [0, -z, -y] : [1, z, -y]; }
  else if (ay >= az) { ma = ay; [face, sc, tc] = y > 0 ? [2, x, z] : [3, x, -z]; }
  else { ma = az; [face, sc, tc] = z > 0 ? [4, x, -y] : [5, -x, -y]; }
  const u = Math.min(n - 1, Math.max(0, Math.floor((sc / ma * 0.5 + 0.5) * n)));
  const v = Math.min(n - 1, Math.max(0, Math.floor((tc / ma * 0.5 + 0.5) * n)));
  return face * n * n + v * n + u;
}

async function gunzip(res: Response): Promise<ArrayBuffer> {
  if (!res.ok || !res.body) throw new Error(`failed to load ${res.url}: ${res.status}`);
  return new Response(res.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

export interface DiffuseSky { size: number; levels: Uint32Array[][] } // [level][face] RGB9E5

export async function loadDiffuse(url: string): Promise<DiffuseSky> {
  const buf = await gunzip(await fetch(url));
  const [size, nLevels] = new Uint32Array(buf, 0, 2);
  const levels: Uint32Array[][] = [];
  let off = 8;
  for (let l = 0, n = size; l < nLevels; l++, n = Math.max(1, n >> 1)) {
    const faces = [];
    for (let f = 0; f < 6; f++) {
      faces.push(new Uint32Array(buf, off, n * n));
      off += n * n * 4;
    }
    levels.push(faces);
  }
  return { size, levels };
}

export interface StarField {
  stars: Float32Array; // vec4 per star: direction, A (= illuminance / Y of unit blackbody) — sorted by cell
  temps: Float32Array;
  cells: Uint32Array; // 6 n^2 + 1 offsets
  coarse: Float16Array[][]; // [level][face] RGBA16F, XYZ radiance * STORE_SCALE
}

export function buildStars(buf: ArrayBuffer): StarField {
  const count = new Uint32Array(buf, 0, 1)[0];
  const raw = new Float32Array(buf, 4, count * 5);
  const n = STAR_CELLS;
  const ids = new Uint32Array(count);
  for (let i = 0; i < count; i++) ids[i] = cellOf(raw[i * 5], raw[i * 5 + 1], raw[i * 5 + 2]);
  const order = Array.from({ length: count }, (_, i) => i).sort((p, q) => ids[p] - ids[q]);
  const cells = new Uint32Array(6 * n * n + 1);
  for (let i = 0; i < count; i++) cells[ids[i] + 1]++;
  for (let c = 0; c < 6 * n * n; c++) cells[c + 1] += cells[c];
  const stars = new Float32Array(count * 4), temps = new Float32Array(count);
  const bb = new Map<number, [number, number, number]>(); // XYZ per whole kelvin (catalogue T is coarse anyway)
  const xyzs: [number, number, number][] = [];
  order.forEach((src, i) => {
    const T = Math.round(bvToTemperature(raw[src * 5 + 4]));
    let c = bb.get(T);
    if (!c) bb.set(T, (c = blackbodyXYZ(T)));
    const A = (LUX_MAG0 * 10 ** (-0.4 * raw[src * 5 + 3])) / c[1]; // XYZ illuminance = A * XYZ_bb(T)
    stars.set([raw[src * 5], raw[src * 5 + 1], raw[src * 5 + 2], A], i * 4);
    temps[i] = T;
    xyzs.push([A * c[0], A * c[1], A * c[2]]);
  });
  return { stars, temps, cells, coarse: coarseCube(stars, xyzs) };
}

/** Catalogue starlight binned into a cube map (radiance = flux / texel solid angle), solid-angle-weighted mips. */
function coarseCube(stars: Float32Array, xyzs: [number, number, number][]): Float16Array[][] {
  const N = COARSE_SIZE;
  const sa = texelSolidAngles(N);
  let level = Array.from({ length: 6 }, () => new Float64Array(N * N * 3));
  for (let i = 0; i < xyzs.length; i++) {
    const c = cellOf(stars[i * 4], stars[i * 4 + 1], stars[i * 4 + 2], N);
    const f = Math.floor(c / (N * N)), t = c % (N * N);
    for (let k = 0; k < 3; k++) level[f][t * 3 + k] += xyzs[i][k] / sa[t];
  }
  const out: Float16Array[][] = [];
  let w = sa, n = N;
  for (;;) {
    out.push(level.map((face) => {
      const h = new Float16Array(n * n * 4);
      for (let t = 0; t < n * n; t++) {
        for (let k = 0; k < 3; k++) h[t * 4 + k] = Math.min(6e4, face[t * 3 + k] * STORE_SCALE);
        h[t * 4 + 3] = 1;
      }
      return h;
    }));
    if (n === 1) break;
    const m = n >> 1;
    const w2 = new Float64Array(m * m);
    const next = level.map(() => new Float64Array(m * m * 3));
    for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
      const kids = [(2 * y) * n + 2 * x, (2 * y) * n + 2 * x + 1, (2 * y + 1) * n + 2 * x, (2 * y + 1) * n + 2 * x + 1];
      w2[y * m + x] = kids.reduce((s, k) => s + w[k], 0);
      for (let f = 0; f < 6; f++) for (let k = 0; k < 3; k++)
        next[f][(y * m + x) * 3 + k] = kids.reduce((s, q) => s + level[f][q * 3 + k] * w[q], 0) / w2[y * m + x];
    }
    level = next;
    w = w2;
    n = m;
  }
  return out;
}

function texelSolidAngles(n: number): Float64Array {
  const F = (x: number, y: number) => Math.atan2(x * y, Math.sqrt(x * x + y * y + 1));
  const out = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x0 = (i / n) * 2 - 1, x1 = ((i + 1) / n) * 2 - 1, y0 = (j / n) * 2 - 1, y1 = ((j + 1) / n) * 2 - 1;
    out[j * n + i] = F(x1, y1) - F(x0, y1) - F(x1, y0) + F(x0, y0);
  }
  return out;
}

/** Smallest angular size of a star cell (at cube corners), used to cap the point-star kernel. */
export const cellMinAngle = (n = STAR_CELLS) => (Math.SQRT2 / 3) * (2 / n);

export async function loadStars(url: string) {
  return buildStars(await gunzip(await fetch(url)));
}

/**
 * Hole frame -> equatorial sky rotation: the galactic centre sits straight behind the hole for an observer on
 * the +x axis, and the spin axis is tilted 25 degrees from the galactic pole (so the Milky Way is not
 * aligned with the disk). Returns the three columns.
 */
export function skyRotation(): number[][] {
  const eq = (raDeg: number, decDeg: number) => {
    const ra = (raDeg * Math.PI) / 180, de = (decDeg * Math.PI) / 180;
    return [Math.cos(de) * Math.cos(ra), Math.cos(de) * Math.sin(ra), Math.sin(de)];
  };
  const gc = eq(266.405, -28.936), ngp = eq(192.859, 27.128);
  const ex = gc.map((v) => -v);
  const tilt = (25 * Math.PI) / 180;
  const side = [ngp[1] * gc[2] - ngp[2] * gc[1], ngp[2] * gc[0] - ngp[0] * gc[2], ngp[0] * gc[1] - ngp[1] * gc[0]];
  const ez = ngp.map((v, i) => Math.cos(tilt) * v + Math.sin(tilt) * side[i]);
  const ey = [ez[1] * ex[2] - ez[2] * ex[1], ez[2] * ex[0] - ez[0] * ex[2], ez[0] * ex[1] - ez[1] * ex[0]];
  return [ex, ey, ez];
}
