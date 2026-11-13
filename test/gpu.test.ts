// The WGSL (f32) tracer against the f64 reference, pixel by pixel, on a headless Dawn device.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { isco, lower, metric, traceRay, type V4 } from "../src/kerr.ts";
import { Observer } from "../src/observer.ts";
import { FLAGS, Renderer, type TraceParams } from "../src/renderer.ts";
import { skyRotation } from "../src/sky.ts";

Object.assign(globalThis, globals);

test("GPU tracer matches the double-precision reference", async (t) => {
  const keepAlive = setInterval(() => {}, 10); // Dawn's promises don't hold Node's event loop open
  t.after(() => clearInterval(keepAlive));
  const gpu = create([]); // must stay referenced: collecting it tears down Dawn under the live device
  const adapter = await gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: any) => errors.push(e.error.message));
  const traceCode = readFileSync(new URL("../src/trace.wgsl", import.meta.url), "utf8");
  const renderer = new Renderer(device, { traceCode, traceFormat: "rgba32float" });
  const W = 80, H = 48; // rows must be 256-byte aligned for the readback
  renderer.resize(W, H);

  const a = 0.9;
  const ob = new Observer({ a, mode: "static", r: 25, theta: 1.35, prograde: true, gyro: false });
  ob.rotate(0.15, 0.05);
  const cam = ob.camera();
  const diskIn = isco(a), diskOut = 18;
  const p: TraceParams = {
    pos: cam.pos, frame: cam.frame, sky: skyRotation(), width: W, height: H, jitter: [0, 0], fovY: 0.9, a, bend: 1,
    diskIn, diskOut, rEscape: 1000, tol: 1e-5, maxSteps: 4000,
    flags: FLAGS.debug | FLAGS.disk | FLAGS.redshift | FLAGS.doppler, skyGain: 1, turbPeriod: 100, inside: false,
  };
  const enc = device.createCommandEncoder();
  renderer.trace(enc, p, 1);
  const bytesPerRow = W * 16;
  const read = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyTextureToBuffer({ texture: renderer.hdr }, { buffer: read, bytesPerRow }, [W, H]);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const px = new Float32Array(read.getMappedRange().slice(0));
  assert.deepEqual(errors, []);

  const g = metric(a, cam.pos);
  const [u, ef, er, eu] = cam.frame;
  const tanHalf = Math.tan(p.fovY / 2), aspect = W / H;
  const cpu = (x: number, y: number) => {
    const ndc = [(2 * x) / W - 1, 1 - (2 * y) / H];
    const n = [1, ndc[0] * tanHalf * aspect, ndc[1] * tanHalf];
    const l = Math.hypot(...n);
    const k = [0, 1, 2, 3].map((m) => u[m] - (n[0] * ef[m] + n[1] * er[m] + n[2] * eu[m]) / l) as V4;
    return traceRay(cam.pos, lower(g, k), { a, rEscape: 1000, diskIn, diskOut, inside: false, tol: 1e-10 });
  };
  // Errors are measured in pixels of image-plane displacement: the GPU result is converted to how far one
  // would have to move across the f64 image to get it (the f64 gradient comes from +-0.05 px samples).
  const kinds = { escape: 0, disk: 0, horizon: 0 };
  let compared = 0, maxDirPx = 0, maxDiskPx = 0, maxG = 0;
  const ang = (p: number[], q: number[]) => // atan2 form: acos(dot) loses everything below ~3e-4 rad
    Math.atan2(Math.hypot(p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]), p[0] * q[0] + p[1] * q[1] + p[2] * q[2]);
  for (let y = 1; y < H; y += 2) {
    for (let x = 1; x < W; x += 2) {
      const ref = cpu(x + 0.5, y + 0.5);
      const nb = [cpu(x + 0.45, y + 0.5), cpu(x + 0.55, y + 0.5), cpu(x + 0.5, y + 0.45), cpu(x + 0.5, y + 0.55)];
      // skip pixels sitting on an image boundary (shadow edge, disk edge, higher-order image edge)
      if (nb.some((e) => e.kind !== ref.kind)) continue;
      const o = (y * W + x) * 4;
      const [v0, v1, v2, v3] = [px[o], px[o + 1], px[o + 2], px[o + 3]];
      const gpuKind = v3 > 0 ? "escape" : v3 < 0 ? "disk" : "horizon";
      const refKind = ref.kind === "maxsteps" ? "horizon" : ref.kind;
      assert.equal(gpuKind, refKind, `pixel ${x},${y}`);
      kinds[refKind]++;
      compared++;
      if (ref.kind === "escape") {
        const d = nb.map((e) => (e.kind === "escape" ? e.dir : ref.dir));
        const perPx = Math.max(ang(d[0], d[1]), ang(d[2], d[3])) / 0.1;
        maxDirPx = Math.max(maxDirPx, ang([v0, v1, v2], ref.dir) / perPx);
        maxG = Math.max(maxG, Math.abs(v3 - ref.g) / ref.g);
      } else if (ref.kind === "disk") {
        const r = nb.map((e) => (e.kind === "disk" ? e.r : ref.r));
        const perPx = Math.max(Math.abs(r[0] - r[1]), Math.abs(r[2] - r[3])) / 0.1;
        maxDiskPx = Math.max(maxDiskPx, Math.abs(v0 - ref.r) / perPx);
        maxG = Math.max(maxG, Math.abs(v1 - ref.g) / ref.g);
      }
    }
  }
  console.log({ compared, kinds, maxDirPx, maxDiskPx, maxG });
  assert.ok(compared > 600 && kinds.escape > 50 && kinds.disk > 50 && kinds.horizon > 20, JSON.stringify(kinds));
  assert.ok(maxDirPx < 0.01, `sky image error ${maxDirPx} px`);
  assert.ok(maxDiskPx < 0.01, `disk image error ${maxDiskPx} px`);
  assert.ok(maxG < 2e-4, `redshift error ${maxG}`);
  device.destroy();
  void gpu;
});
