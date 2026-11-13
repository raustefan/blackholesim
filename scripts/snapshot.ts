// Render a plate headless (Dawn) through the full pipeline and write a PNG. Dev tool:
//   node scripts/snapshot.ts out.png [json overrides]
// e.g. node scripts/snapshot.ts /tmp/a.png '{"r":8,"theta":85,"a":0.99}'
import { readFileSync, writeFileSync } from "node:fs";
import { deflateSync } from "node:zlib";
import { create, globals } from "webgpu";
import { isco, horizonOuter, diskTemperature } from "../src/kerr.ts";
import { Observer, type Mode } from "../src/observer.ts";
import { DISK_N, FLAGS, Renderer } from "../src/renderer.ts";
import { loadDiffuse, loadStars, skyRotation } from "../src/sky.ts";

Object.assign(globalThis, globals);
const root = new URL("..", import.meta.url);
globalThis.fetch = (async (u: string) => new Response(readFileSync(new URL(`public${u}`, root)))) as typeof fetch;

const o = {
  a: 0.9, M: 6.5e9, mdot: 1e-4, rOut: 30, r: 30, theta: 82, mode: "static" as Mode, fov: 60, skyGain: 36, ev: 0,
  yaw: 0, pitch: 0, w: 1280, h: 800, flags: FLAGS.redshift | FLAGS.doppler | FLAGS.disk | FLAGS.stars | FLAGS.turbulence,
  bend: 1, tau: 0, tol: 1e-5, frames: 12,
  ...JSON.parse(process.argv[3] ?? "{}"),
};

const keepAlive = setInterval(() => {}, 50);
const gpu = create([]);
const device = await (await gpu.requestAdapter())!.requestDevice();
device.addEventListener("uncapturederror", (e: any) => console.error(e.error.message));
const [diffuse, stars] = await Promise.all([loadDiffuse("/sky-diffuse.bin"), loadStars("/stars.bin")]);
const rd = new Renderer(device, {
  traceCode: readFileSync(new URL("src/trace.wgsl", root), "utf8"),
  postCode: readFileSync(new URL("src/post.wgsl", root), "utf8"),
  canvasFormat: "rgba8unorm", diffuse, stars,
});
rd.resize(o.w, o.h);
const dIn = isco(o.a);
const T = new Float32Array(DISK_N).map((_, i) => diskTemperature(o.a, dIn * (o.rOut / dIn) ** (i / (DISK_N - 1)), o.M, o.mdot));
rd.setDisk(T);

const ob = new Observer({ a: o.a, mode: o.mode === "free" ? "static" : o.mode, r: o.r, theta: (o.theta * Math.PI) / 180, prograde: true, gyro: false });
if (o.mode === "free") { ob.release(); ob.advance(o.tau); }
ob.rotate(o.yaw * Math.PI / 180, o.pitch * Math.PI / 180);
const cam = ob.camera();
const out = device.createTexture({ size: [o.w, o.h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const t0 = performance.now();
for (let f = 0; f < o.frames; f++) {
  const enc = device.createCommandEncoder();
  rd.trace(enc, {
    pos: cam.pos, frame: o.bend ? cam.frame : ob.flatFrame(), sky: skyRotation(), width: o.w, height: o.h, jitter: [0, 0],
    fovY: (o.fov * Math.PI) / 180, a: o.a, bend: o.bend, diskIn: dIn, diskOut: o.rOut, rEscape: Math.max(1000, 2 * ob.rNow), tol: o.tol,
    maxSteps: 3000, flags: o.flags, skyGain: 2 ** o.skyGain, turbPeriod: 150, inside: ob.rNow < horizonOuter(o.a),
  }, 1);
  rd.post(enc, out.createView(), { w: o.w, h: o.h, dt: 1, evComp: o.ev, bloom: 0.045, grain: 0.035, vignette: 0.32, luminet: !!(o.flags & FLAGS.luminet), time: f });
  device.queue.submit([enc.finish()]);
}
const rb = device.createBuffer({ size: o.w * o.h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const enc = device.createCommandEncoder();
enc.copyTextureToBuffer({ texture: out }, { buffer: rb, bytesPerRow: o.w * 4 }, [o.w, o.h]);
device.queue.submit([enc.finish()]);
await rb.mapAsync(GPUMapMode.READ);
const px = new Uint8Array(rb.getMappedRange().slice(0));
console.log(`${o.frames} frames in ${(performance.now() - t0).toFixed(0)} ms (incl. GPU), r = ${ob.rNow.toFixed(3)}`);

// minimal PNG writer
const crcTable = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
const crc = (b: Uint8Array) => { let c = 0xffffffff; for (const x of b) c = crcTable[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
const chunk = (type: string, data: Uint8Array) => {
  const b = new Uint8Array(12 + data.length), v = new DataView(b.buffer);
  v.setUint32(0, data.length);
  b.set(new TextEncoder().encode(type), 4);
  b.set(data, 8);
  v.setUint32(8 + data.length, crc(b.subarray(4, 8 + data.length)));
  return b;
};
const ihdr = new Uint8Array(13), hv = new DataView(ihdr.buffer);
hv.setUint32(0, o.w); hv.setUint32(4, o.h); ihdr.set([8, 6, 0, 0, 0], 8);
const raw = new Uint8Array(o.h * (o.w * 4 + 1));
for (let y = 0; y < o.h; y++) raw.set(px.subarray(y * o.w * 4, (y + 1) * o.w * 4), y * (o.w * 4 + 1) + 1);
writeFileSync(process.argv[2], Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", new Uint8Array())]));
device.destroy();
clearInterval(keepAlive);
void gpu;
