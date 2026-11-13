// Blackbody colorimetry in absolute photometric units.
// A blackbody of temperature T seen with frequency shift g looks exactly like a blackbody of temperature g T
// (I_nu / nu^3 is invariant), so one table of XYZ(T) serves the disk, the stars and the redshifted sky.
import { CIE_START_NM, CIE_STEP_NM, CIE_XYZ } from "./cie1931.ts";

const h = 6.62607015e-34, c = 299792458, kB = 1.380649e-23;

/** Spectral radiance B_lambda [W m^-2 sr^-1 m^-1]. */
export function planck(lambdaM: number, T: number): number {
  const x = (h * c) / (lambdaM * kB * T);
  if (x > 700) return 0;
  return (2 * h * c * c) / lambdaM ** 5 / Math.expm1(x);
}

/** CIE XYZ of a blackbody [cd m^-2] (Y is luminance): 683 lm/W * int B_lambda cmf dlambda (trapezoid, 5 nm). */
export function blackbodyXYZ(T: number): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  const n = CIE_XYZ.length;
  for (let i = 0; i < n; i++) {
    const w = (i === 0 || i === n - 1 ? 0.5 : 1) * CIE_STEP_NM * 1e-9;
    const B = planck((CIE_START_NM + i * CIE_STEP_NM) * 1e-9, T);
    for (let k = 0; k < 3; k++) out[k] += 683 * B * CIE_XYZ[i][k] * w;
  }
  return out;
}

/** XYZ -> linear sRGB (Rec. 709 primaries, D65). */
export const XYZ_TO_SRGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.969266, 1.8760108, 0.041556],
  [0.0556434, -0.2040259, 1.0572252],
];

export const LUT_LOG10_T0 = 2.5;
export const LUT_LOG10_T1 = 9.5;
export const LUT_SIZE = 1024;

/** ln X, ln Y, ln Z of a blackbody, uniformly in log10 T (clamped at e^-80 so it fits f32). */
export function blackbodyLUT(): Float32Array {
  const out = new Float32Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const T = 10 ** (LUT_LOG10_T0 + ((LUT_LOG10_T1 - LUT_LOG10_T0) * i) / (LUT_SIZE - 1));
    const xyz = blackbodyXYZ(T);
    for (let k = 0; k < 3; k++) out[i * 4 + k] = Math.max(-80, Math.log(xyz[k]));
  }
  return out;
}

/** [x0, x1, then 256 Planckian temperatures at uniform CIE x in [x0, x1]] for 1000 K .. 1e6 K (x falls with T). */
export function chromaticityToTemperature(): Float32Array {
  const xOf = (T: number) => { const [X, Y, Z] = blackbodyXYZ(T); return X / (X + Y + Z); };
  const Ts = Array.from({ length: 2000 }, (_, i) => 1000 * 1000 ** (i / 1999));
  const xs = Ts.map(xOf);
  const x0 = xs[xs.length - 1], x1 = xs[0];
  const out = new Float32Array(258);
  out[0] = x0;
  out[1] = x1;
  for (let i = 0; i < 256; i++) {
    const x = x0 + ((x1 - x0) * i) / 255;
    let j = xs.findIndex((v) => v <= x); // xs is decreasing
    j = Math.max(1, j);
    const f = (xs[j - 1] - x) / (xs[j - 1] - xs[j]);
    out[2 + i] = Ts[j - 1] * (Ts[j] / Ts[j - 1]) ** f;
  }
  return out;
}

/** Effective temperature from B-V colour index (Ballesteros 2012, EPL 97, 34008). */
export const bvToTemperature = (bv: number) => {
  const x = Math.min(2.0, Math.max(-0.4, bv));
  return 4600 * (1 / (0.92 * x + 1.7) + 1 / (0.92 * x + 0.62));
};

export const LUX_MAG0 = 2.54e-6; // illuminance of a V = 0 star [lx]
