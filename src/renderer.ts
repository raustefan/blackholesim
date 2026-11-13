// WebGPU frame graph: trace (with progressive accumulation) -> exposure meter -> bloom -> AgX composite.
import { blackbodyLUT, chromaticityToTemperature } from "./color.ts";
import { COARSE_SIZE, STAR_CELLS, cellMinAngle, type DiffuseSky, type StarField } from "./sky.ts";
import type { V4 } from "./kerr.ts";

export const FLAGS = { doppler: 1, redshift: 2, luminet: 4, turbulence: 8, stars: 16, disk: 32, debug: 64 };
export const DISK_N = 512;
const BLOOM_LEVELS = 6;

export interface TraceParams {
  pos: V4;
  frame: V4[]; // u, forward, right, up
  sky: number[][]; // columns
  width: number;
  height: number;
  jitter: [number, number];
  fovY: number;
  a: number;
  bend: number;
  diskIn: number;
  diskOut: number;
  rEscape: number;
  tol: number;
  maxSteps: number;
  flags: number;
  skyGain: number;
  turbPeriod: number;
  inside: boolean;
}

export function packTrace(p: TraceParams): ArrayBuffer {
  const buf = new ArrayBuffer(208);
  const f = new Float32Array(buf), u = new Uint32Array(buf);
  f.set(p.pos, 0);
  p.frame.forEach((e, i) => f.set(e, 4 + 4 * i));
  p.sky.forEach((c, i) => f.set(c, 20 + 4 * i));
  f.set([p.width, p.height, p.jitter[0], p.jitter[1]], 32);
  f.set([Math.tan(p.fovY / 2), p.width / p.height, p.a, p.bend, p.diskIn, p.diskOut, p.rEscape, p.tol], 36);
  u[44] = p.maxSteps;
  u[45] = p.flags;
  f[46] = p.skyGain;
  f[47] = p.turbPeriod;
  u[48] = STAR_CELLS;
  f[49] = cellMinAngle();
  u[50] = p.inside ? 1 : 0;
  return buf;
}

const storage = (device: GPUDevice, data: ArrayBufferView, extra = 0) => {
  const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
  device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
  return b;
};

export interface RendererOpts {
  traceCode: string;
  postCode?: string; // omitted: trace-only (tests)
  canvasFormat?: GPUTextureFormat;
  traceFormat?: GPUTextureFormat;
  diffuse?: DiffuseSky;
  stars?: StarField;
}

export class Renderer {
  device: GPUDevice;
  o: RendererOpts;
  traceFormat: GPUTextureFormat;
  tracePipe: GPURenderPipeline;
  traceUniform: GPUBuffer;
  diskBuf: GPUBuffer;
  expoBuf: GPUBuffer;
  traceBind: GPUBindGroup;
  hdr!: GPUTexture;
  width = 0;
  height = 0;
  // post
  histPipe?: GPUComputePipeline;
  adaptPipe?: GPUComputePipeline;
  downPipe?: GPURenderPipeline;
  upPipe?: GPURenderPipeline;
  compPipe?: GPURenderPipeline;
  histBuf?: GPUBuffer;
  expoUniform?: GPUBuffer;
  compUniform?: GPUBuffer;
  bloomTex: GPUTexture[] = [];
  bloomUniforms: GPUBuffer[] = [];
  lin: GPUSampler;

  constructor(device: GPUDevice, o: RendererOpts) {
    this.device = device;
    this.o = o;
    this.traceFormat = o.traceFormat ?? "rgba16float";
    const traceModule = device.createShaderModule({ code: o.traceCode });
    const blendable = this.traceFormat === "rgba16float";
    this.tracePipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: traceModule, entryPoint: "vs" },
      fragment: {
        module: traceModule,
        entryPoint: "fs",
        targets: [{
          format: this.traceFormat,
          // progressive refinement: new = constant * sample + (1 - constant) * history
          ...(blendable && { blend: { color: { srcFactor: "constant", dstFactor: "one-minus-constant" }, alpha: { srcFactor: "one", dstFactor: "zero" } } }),
        }],
      },
    });
    this.traceUniform = device.createBuffer({ size: 208, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.diskBuf = storage(device, new Float32Array(DISK_N));
    this.expoBuf = storage(device, new Float32Array([1e-6, 0, 0, 0]), GPUBufferUsage.COPY_SRC);
    this.lin = device.createSampler({
      magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", maxAnisotropy: 16,
      addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
    });

    const cube = (size: number, format: GPUTextureFormat, mips: number) =>
      device.createTexture({ size: [size, size, 6], format, mipLevelCount: mips, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    let diffuseTex: GPUTexture;
    if (o.diffuse) {
      diffuseTex = cube(o.diffuse.size, "rgb9e5ufloat", o.diffuse.levels.length);
      o.diffuse.levels.forEach((faces, l) => {
        const n = Math.max(1, o.diffuse!.size >> l);
        faces.forEach((data, f) => device.queue.writeTexture({ texture: diffuseTex, mipLevel: l, origin: [0, 0, f] }, data, { bytesPerRow: n * 4 }, [n, n, 1]));
      });
    } else diffuseTex = cube(1, "rgb9e5ufloat", 1);
    let coarseTex: GPUTexture;
    if (o.stars) {
      coarseTex = cube(COARSE_SIZE, "rgba16float", o.stars.coarse.length);
      o.stars.coarse.forEach((faces, l) => {
        const n = Math.max(1, COARSE_SIZE >> l);
        faces.forEach((data, f) => device.queue.writeTexture({ texture: coarseTex, mipLevel: l, origin: [0, 0, f] }, new Uint8Array(data.buffer, data.byteOffset, data.byteLength), { bytesPerRow: n * 8 }, [n, n, 1]));
      });
    } else coarseTex = cube(1, "rgba16float", 1);
    const stars = o.stars ?? { stars: new Float32Array(4), temps: new Float32Array(1), cells: new Uint32Array(6 * STAR_CELLS * STAR_CELLS + 1) };

    this.traceBind = device.createBindGroup({
      layout: this.tracePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.traceUniform } },
        { binding: 1, resource: { buffer: storage(device, blackbodyLUT()) } },
        { binding: 2, resource: { buffer: this.diskBuf } },
        { binding: 3, resource: diffuseTex.createView({ dimension: "cube" }) },
        { binding: 4, resource: coarseTex.createView({ dimension: "cube" }) },
        { binding: 5, resource: this.lin },
        { binding: 6, resource: { buffer: storage(device, stars.stars) } },
        { binding: 7, resource: { buffer: storage(device, stars.temps) } },
        { binding: 8, resource: { buffer: storage(device, stars.cells) } },
        { binding: 9, resource: { buffer: this.expoBuf } },
        { binding: 10, resource: { buffer: storage(device, chromaticityToTemperature()) } },
      ],
    });

    if (o.postCode) {
      const pm = device.createShaderModule({ code: o.postCode });
      this.histPipe = device.createComputePipeline({ layout: "auto", compute: { module: pm, entryPoint: "histogram" } });
      this.adaptPipe = device.createComputePipeline({ layout: "auto", compute: { module: pm, entryPoint: "adapt" } });
      const full = (entryPoint: string, format: GPUTextureFormat, blend?: GPUBlendState) => device.createRenderPipeline({
        layout: "auto",
        vertex: { module: pm, entryPoint: "vsFull" },
        fragment: { module: pm, entryPoint, targets: [{ format, blend }] },
      });
      this.downPipe = full("bloomDown", "rgba16float");
      this.upPipe = full("bloomUp", "rgba16float", { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } });
      this.compPipe = full("composite", o.canvasFormat ?? "bgra8unorm");
      this.histBuf = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.STORAGE });
      this.expoUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.compUniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      for (let i = 0; i < 2 * BLOOM_LEVELS; i++)
        this.bloomUniforms.push(device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
  }

  setDisk(temps: Float32Array) {
    this.device.queue.writeBuffer(this.diskBuf, 0, temps.buffer, temps.byteOffset, temps.byteLength);
  }

  /** Allocate full-resolution targets; frames may trace into a smaller top-left sub-rectangle. */
  resize(width: number, height: number) {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.hdr?.destroy();
    this.hdr = this.device.createTexture({
      size: [width, height],
      format: this.traceFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.bloomTex.forEach((t) => t.destroy());
    this.bloomTex = [];
    if (this.o.postCode)
      for (let i = 0; i < BLOOM_LEVELS; i++)
        this.bloomTex.push(this.device.createTexture({
          size: [Math.max(1, width >> (i + 1)), Math.max(1, height >> (i + 1))],
          format: "rgba16float",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        }));
  }

  /** Trace into the HDR target's top-left (p.width x p.height); blend = 1 replaces, 1/n accumulates. */
  trace(enc: GPUCommandEncoder, p: TraceParams, blend: number) {
    this.device.queue.writeBuffer(this.traceUniform, 0, packTrace(p));
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.hdr.createView(), loadOp: "load", storeOp: "store" }] });
    pass.setPipeline(this.tracePipe);
    pass.setBindGroup(0, this.traceBind);
    pass.setViewport(0, 0, p.width, p.height, 0, 1);
    pass.setScissorRect(0, 0, p.width, p.height);
    if (this.traceFormat === "rgba16float") pass.setBlendConstant([blend, blend, blend, blend]);
    pass.draw(3);
    pass.end();
  }

  post(enc: GPUCommandEncoder, target: GPUTextureView, p: { w: number; h: number; dt: number; evComp: number; bloom: number; grain: number; vignette: number; luminet: boolean; time: number }) {
    const d = this.device;
    const sx = p.w / this.width, sy = p.h / this.height;
    // exposure
    const eu = new ArrayBuffer(32);
    new Uint32Array(eu).set([p.w, p.h]);
    new Float32Array(eu).set([p.dt, p.evComp, -40, 64, 2.5], 2);
    d.queue.writeBuffer(this.expoUniform!, 0, eu);
    const cp = enc.beginComputePass();
    cp.setPipeline(this.histPipe!);
    cp.setBindGroup(0, d.createBindGroup({
      layout: this.histPipe!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.hdr.createView() },
        { binding: 1, resource: { buffer: this.histBuf! } },
        { binding: 2, resource: { buffer: this.expoBuf } },
        { binding: 3, resource: { buffer: this.expoUniform! } },
      ],
    }));
    cp.dispatchWorkgroups(Math.ceil(p.w / 16), Math.ceil(p.h / 16));
    cp.setPipeline(this.adaptPipe!);
    cp.setBindGroup(0, d.createBindGroup({
      layout: this.adaptPipe!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.histBuf! } },
        { binding: 2, resource: { buffer: this.expoBuf } },
        { binding: 3, resource: { buffer: this.expoUniform! } },
      ],
    }));
    cp.dispatchWorkgroups(1);
    cp.end();

    // bloom: downsample chain then additive tent upsample
    const blit = (pipe: GPURenderPipeline, src: GPUTexture, dst: GPUTexture, ub: GPUBuffer, u: number[], load: GPULoadOp) => {
      d.queue.writeBuffer(ub, 0, new Float32Array(u));
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: dst.createView(), loadOp: load, clearValue: [0, 0, 0, 0], storeOp: "store" }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: src.createView() }, { binding: 1, resource: this.lin }, { binding: 2, resource: { buffer: ub } }],
      }));
      pass.draw(3);
      pass.end();
    };
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      const src = i === 0 ? this.hdr : this.bloomTex[i - 1];
      blit(this.downPipe!, src, this.bloomTex[i], this.bloomUniforms[i],
        [1 / src.width, 1 / src.height, i === 0 ? sx : 1, i === 0 ? sy : 1, i === 0 ? 1 : 0, 1, 0, 0], "clear");
    }
    for (let i = BLOOM_LEVELS - 1; i > 0; i--) {
      const src = this.bloomTex[i];
      blit(this.upPipe!, src, this.bloomTex[i - 1], this.bloomUniforms[BLOOM_LEVELS + i], [1 / src.width, 1 / src.height, 1, 1, 0, 1, 0, 0], "load");
    }

    const cu = new ArrayBuffer(48);
    new Float32Array(cu).set([sx, sy, this.width, this.height, p.bloom, p.time, p.grain, p.vignette]);
    new Uint32Array(cu)[8] = p.luminet ? 1 : 0;
    new Float32Array(cu)[9] = BLOOM_LEVELS;
    d.queue.writeBuffer(this.compUniform!, 0, cu);
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: "clear", clearValue: [0, 0, 0, 1], storeOp: "store" }] });
    pass.setPipeline(this.compPipe!);
    pass.setBindGroup(0, d.createBindGroup({
      layout: this.compPipe!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.hdr.createView() },
        { binding: 1, resource: this.bloomTex[0].createView() },
        { binding: 2, resource: this.lin },
        { binding: 3, resource: { buffer: this.compUniform! } },
      ],
    }));
    pass.draw(3);
    pass.end();
  }
}
