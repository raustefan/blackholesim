"""Build static data for the simulator (run once; outputs are committed).

  python -I scripts/prepare_data.py <starmap_2020_4k.exr> <hygdata_v41.csv>

Inputs
  * NASA SVS Deep Star Maps 2020, equatorial, linear EXR   https://svs.gsfc.nasa.gov/4851
  * HYG star catalogue v4.1 (Hipparcos + Yale + Gliese)     https://github.com/astronexus/HYG-Database
  * CIE 1931 2-degree colour matching functions (official CIE data, via colour-science)

Point stars come from the catalogue (true point sources with V magnitudes). The NASA map's stellar
brightnesses are display-compressed (aperture photometry vs. catalogue magnitudes is far from linear), so
the map is only used for the *diffuse* glow: catalogue stars are cut out, remaining point sources are
median-filtered away, and the result is calibrated to the integrated starlight at the galactic poles
(Leinert et al. 1998, A&AS 127, 1: ~40 S10(V) units; uncertain to ~factor 2).

Outputs
  * public/sky-diffuse.bin  gzip( u32 size, u32 levels, then per level 6 faces of size^2 RGB9E5 )
                            cube faces in WebGPU order +X,-X,+Y,-Y,+Z,-Z, equatorial frame
                            (x: RA 0h, z: north celestial pole); units cd m^-2 * STORE_SCALE
  * public/stars.bin        gzip( u32 count, then count * f32[x, y, z, V magnitude, B-V] )
  * src/cie1931.ts          CMFs at 5 nm, 360..830 nm
Needs: numpy, OpenEXR, colour-science.
"""
import csv, gzip, struct, sys
from pathlib import Path
import numpy as np
import OpenEXR
import colour

root = Path(__file__).resolve().parent.parent
CUBE = 512
LUX_MAG0 = 2.54e-6  # illuminance of a V = 0 star [lx]
S10_PER_SR = 1e-4 / np.radians(1) ** 2  # one V=10 star per square degree, in "V=0 stars per sr"
POLE_S10 = 40.0
STORE_SCALE = 1e4  # textures hold cd/m^2 * 1e4 so faint sky sits mid-range in RGB9E5's exponent


def rgb9e5(rgb):
    """Pack float RGB (...,3) into uint32 per the WebGPU/GL shared-exponent spec."""
    N, B, EMAX = 9, 15, 31
    c = np.clip(np.nan_to_num(rgb.astype(np.float64)), 0, (2**N - 1) / 2**N * 2.0 ** (EMAX - B))
    maxc = c.max(axis=-1)
    e = np.maximum(-B - 1, np.floor(np.log2(np.maximum(maxc, 1e-300)))) + 1 + B
    denom = 2.0 ** (e - B - N)
    bump = np.floor(maxc / denom + 0.5) == 2**N
    denom = np.where(bump, denom * 2, denom)
    e = np.where(bump, e + 1, e)
    m = np.floor(c / denom[..., None] + 0.5).astype(np.uint32)
    return (m[..., 0] | (m[..., 1] << 9) | (m[..., 2] << 18) | (e.astype(np.uint32) << 27)).astype("<u4")


def cube_dirs(n):
    """Unit directions of texel centres, shape (6, n, n, 3), WebGPU cube conventions."""
    a = (np.arange(n) + 0.5) / n * 2 - 1
    s, t = np.meshgrid(a, a)  # s along columns, t along rows (down)
    one = np.ones_like(s)
    faces = [(one, -t, -s), (-one, -t, s), (s, one, t), (s, -one, -t), (s, -t, one), (-s, -t, -one)]
    d = np.stack([np.stack(f, -1) for f in faces])
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def texel_solid_angle(n):
    a = np.arange(n + 1) / n * 2 - 1
    x, y = np.meshgrid(a, a)
    f = np.arctan2(x * y, np.sqrt(x * x + y * y + 1))  # area integral of the cube-face projection
    return f[1:, 1:] - f[:-1, 1:] - f[1:, :-1] + f[:-1, :-1]


def load_catalogue(path):
    rows = [r for r in csv.DictReader(open(path)) if r["proper"] != "Sol"]
    ra = np.array([float(r["rarad"]) for r in rows])
    dec = np.array([float(r["decrad"]) for r in rows])
    mag = np.array([float(r["mag"]) for r in rows])
    bv = np.array([float(r["ci"]) if r["ci"] else 0.65 for r in rows])  # 33 stars lack B-V: solar
    return ra, dec, mag, bv


def median5(img):
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="edge")
    p[:, :2], p[:, -2:] = p[:, -4:-2], p[:, 2:4]  # wrap in RA
    out = np.empty_like(img)
    for y in range(0, img.shape[0], 128):  # row blocks keep memory bounded
        blk = p[y : y + 128 + 4]
        out[y : y + 128] = np.median(sliding_window_view(blk, (5, 5), axis=(0, 1)), axis=(-2, -1))
    return out


def diffuse(exr, ra, dec, mag):
    img = OpenEXR.File(exr).channels()["RGB"].pixels.astype(np.float32)
    H, W, _ = img.shape
    x = ((0.5 - ra / (2 * np.pi)) % 1) * W
    y = (0.5 - dec / np.pi) * H
    # Cut out catalogue stars (radius grows with brightness: halo of saturated stars) and fill with the
    # median of a surrounding annulus.
    for cx, cy, m in zip(x, y, mag):
        r = 2 if m > 7 else 3 if m > 5 else 5 if m > 3 else 8 if m > 1 else 12
        ys, xs = np.mgrid[-r - 3 : r + 4, -r - 3 : r + 4]
        rr = np.hypot(xs, ys)
        yy, xx = np.clip(int(cy) + ys, 0, H - 1), (int(cx) + xs) % W
        img[yy[rr <= r], xx[rr <= r]] = np.median(img[yy[rr > r], xx[rr > r]], axis=0)
    img = median5(img).astype(np.float32)  # remaining (fainter, map-only) point sources
    # Resample into a cube: 4x4 supersampling per texel, bilinear in the equirect map.
    n, ss = CUBE, 4
    acc = np.zeros((6, n, n, 3))
    for oy in range(ss):
        for ox in range(ss):
            a = (np.arange(n) + (np.array([ox, oy]) + 0.5)[:, None] / ss) / n * 2 - 1
            s, t = np.meshgrid(a[0], a[1])
            one = np.ones_like(s)
            faces = [(one, -t, -s), (-one, -t, s), (s, one, t), (s, -one, -t), (s, -t, one), (-s, -t, -one)]
            d = np.stack([np.stack(f, -1) for f in faces])
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            u = (0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi)) % 1 * W - 0.5
            v = (0.5 - np.arcsin(np.clip(d[..., 2], -1, 1)) / np.pi) * H - 0.5
            u0, v0 = np.floor(u).astype(int), np.clip(np.floor(v).astype(int), 0, H - 2)
            fu, fv = (u - u0)[..., None], np.clip(v - v0, 0, 1)[..., None]
            u0, u1 = u0 % W, (u0 + 1) % W
            acc += (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u1] * fu * (1 - fv)
                    + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u1] * fu * fv)
    cube = acc / ss**2
    # Calibrate: galactic-pole luminance := POLE_S10 S10 units (in cd/m^2).
    d = cube_dirs(n)
    ngp = np.array([np.cos(np.radians(27.128)) * np.cos(np.radians(192.859)),
                    np.cos(np.radians(27.128)) * np.sin(np.radians(192.859)), np.sin(np.radians(27.128))])
    sinb = np.abs(d @ ngp)
    lum = cube @ np.array([0.2126, 0.7152, 0.0722])
    w = texel_solid_angle(n)[None] * (sinb > np.sin(np.radians(70)))
    scale = POLE_S10 * S10_PER_SR * LUX_MAG0 / ((lum * w).sum() / w.sum())
    cube *= scale
    print(f"diffuse: pole calibration scale {scale:.3e}, max {cube.max():.3e} cd/m^2")
    cube *= STORE_SCALE
    # Mip chain with solid-angle weighted averaging.
    levels, sa = [cube], texel_solid_angle(n)[None, ..., None]
    while levels[-1].shape[1] > 1:
        c, k = levels[-1], levels[-1].shape[1] // 2
        wsum = (sa).reshape(1, k, 2, k, 2, 1).sum(axis=(2, 4))
        levels.append((c * sa).reshape(6, k, 2, k, 2, 3).sum(axis=(2, 4)) / wsum)
        sa = wsum
    data = struct.pack("<II", n, len(levels)) + b"".join(rgb9e5(lv).tobytes() for lv in levels)
    (root / "public" / "sky-diffuse.bin").write_bytes(gzip.compress(data, 9))
    return cube


def stars(ra, dec, mag, bv):
    xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec), mag, bv], -1)
    data = struct.pack("<I", len(mag)) + xyz.astype("<f4").tobytes()
    (root / "public" / "stars.bin").write_bytes(gzip.compress(data, 9))
    print(f"stars: {len(mag)}")


def cie():
    cmf = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl = np.arange(360, 831, 5)
    rows = ",\n".join(f"  [{c[0]:.6e}, {c[1]:.6e}, {c[2]:.6e}]" for c in (cmf[w] for w in wl))
    (root / "src" / "cie1931.ts").write_text(
        "// CIE 1931 2-degree standard observer colour matching functions (official CIE data),\n"
        "// sampled every 5 nm from 360 nm to 830 nm. Generated by scripts/prepare_data.py.\n"
        "export const CIE_START_NM = 360;\nexport const CIE_STEP_NM = 5;\n"
        f"export const CIE_XYZ: readonly (readonly [number, number, number])[] = [\n{rows},\n];\n"
    )


if __name__ == "__main__":
    (root / "public").mkdir(exist_ok=True)
    cie()
    ra, dec, mag, bv = load_catalogue(sys.argv[2])
    stars(ra, dec, mag, bv)
    cube = diffuse(sys.argv[1], ra, dec, mag)
    if len(sys.argv) > 3:  # optional preview PNG of the +Z... faces strip, for eyeballing
        import imageio.v3 as iio
        strip = np.concatenate(list(cube), axis=1)
        iio.imwrite(sys.argv[3], (np.clip((strip / np.percentile(strip, 99.5)) ** (1 / 2.2), 0, 1) * 255).astype(np.uint8))
