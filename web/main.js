import init, { default_width, default_height, scene_bytes } from "./pkg/ray_tracer.js";

const canvas = document.getElementById("screen");
const meta = document.getElementById("meta");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });

if (!ctx) {
  throw new Error("Canvas 2D context not available");
}

await init();

if (!navigator.gpu) {
  throw new Error("WebGPU is not available in this browser");
}

const width = default_width();
const height = default_height();
canvas.width = width;
canvas.height = height;

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No WebGPU adapter found");
}

const device = await adapter.requestDevice();
const shaderCode = await (await fetch("../src/trace.wgsl")).text();
const shader = device.createShaderModule({
  label: "Tracing Shader",
  code: shaderCode,
});

const pipeline = device.createComputePipeline({
  label: "Compute Pipeline",
  layout: "auto",
  compute: {
    module: shader,
    entryPoint: "main",
  },
});

const scene = scene_bytes();
const sceneBuffer = device.createBuffer({
  label: "Input Buffer",
  size: scene.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});
new Uint8Array(sceneBuffer.getMappedRange()).set(scene);
sceneBuffer.unmap();

const materials = createMaterialsBytes();
const materialsBuffer = device.createBuffer({
  label: "Materials Buffer",
  size: materials.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});
new Uint8Array(materialsBuffer.getMappedRange()).set(materials);
materialsBuffer.unmap();

const outputSize = width * height * 4;
const outputBuffer = device.createBuffer({
  label: "Output Buffer",
  size: outputSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const stagingBuffer = device.createBuffer({
  label: "Staging Buffer",
  size: outputSize,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const paramsBuffer = device.createBuffer({
  label: "Params Buffer",
  size: 96,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: sceneBuffer } },
    { binding: 1, resource: { buffer: outputBuffer } },
    { binding: 2, resource: { buffer: paramsBuffer } },
    { binding: 3, resource: { buffer: materialsBuffer } },
  ],
});

const suzanneCenter = computeSuzanneCenter(scene);
const rgba = new Uint8ClampedArray(width * height * 4);
const imageData = new ImageData(rgba, width, height);

let frame = 0;
let frameCount = 0;
let fpsTimer = performance.now();

async function tick(now) {
  const camera = orbitCamera(suzanneCenter, now * 0.001, width / height);
  const params = buildParams(width, height, camera, 4, 8, frame);
  device.queue.writeBuffer(paramsBuffer, 0, params);

  const encoder = device.createCommandEncoder({ label: "Compute Encoder" });
  const pass = encoder.beginComputePass({ label: "Compute Pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
  pass.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint32Array(stagingBuffer.getMappedRange());
  for (let i = 0; i < mapped.length; i += 1) {
    const pixel = mapped[i];
    const j = i * 4;
    rgba[j] = (pixel >>> 16) & 0xff;
    rgba[j + 1] = (pixel >>> 8) & 0xff;
    rgba[j + 2] = pixel & 0xff;
    rgba[j + 3] = 255;
  }
  stagingBuffer.unmap();

  ctx.putImageData(imageData, 0, 0);

  frame += 1;
  frameCount += 1;
  const elapsed = now - fpsTimer;
  if (elapsed >= 1000) {
    const fps = (frameCount * 1000) / elapsed;
    document.title = `Ray Tracer Wasm - ${fps.toFixed(1)} FPS`;
    meta.textContent = `shader trace.wgsl   canvas ${width}x${height}   fps ${fps.toFixed(1)}`;
    frameCount = 0;
    fpsTimer = now;
  }

  requestAnimationFrame((t) => {
    tick(t).catch((err) => {
      console.error(err);
      meta.textContent = `render error: ${String(err)}`;
    });
  });
}

requestAnimationFrame((t) => {
  tick(t).catch((err) => {
    console.error(err);
    meta.textContent = `startup error: ${String(err)}`;
  });
});

function createMaterialsBytes() {
  const defs = [
    { emission: [0.0, 0.0, 0.0, 0.0], albedo: [0.5, 0.5, 0.5, 0.0], type: 1 },
    { emission: [0.0, 0.0, 0.0, 0.0], albedo: [0.99, 0.2, 0.2, 0.0], type: 1 },
    { emission: [0.0, 0.0, 0.0, 0.0], albedo: [0.2, 0.2, 0.99, 0.0], type: 1 },
    { emission: [0.0, 0.0, 0.0, 0.0], albedo: [0.2, 0.99, 0.2, 0.0], type: 1 },
    { emission: [0.0, 0.0, 0.0, 0.0], albedo: [0.99, 0.99, 0.99, 0.0], type: 1 },
    { emission: [0.99, 0.99, 0.99, 0.99], albedo: [0.99, 0.99, 0.99, 0.0], type: 1 },
  ];

  const stride = 48;
  const bytes = new ArrayBuffer(defs.length * stride);
  const view = new DataView(bytes);

  defs.forEach((m, idx) => {
    let o = idx * stride;
    m.emission.forEach((v) => {
      view.setFloat32(o, v, true);
      o += 4;
    });
    m.albedo.forEach((v) => {
      view.setFloat32(o, v, true);
      o += 4;
    });
    view.setUint32(o, m.type, true);
    o += 4;
    view.setUint32(o, 0, true);
    o += 4;
    view.setUint32(o, 0, true);
    o += 4;
    view.setUint32(o, 0, true);
  });

  return new Uint8Array(bytes);
}

function buildParams(widthPx, heightPx, camera, depth, samples, frame) {
  const buffer = new ArrayBuffer(96);
  const view = new DataView(buffer);

  view.setUint32(0, widthPx, true);
  view.setUint32(4, heightPx, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);

  writeVec4(view, 16, camera.origin);
  writeVec4(view, 32, camera.lowerLeftCorner);
  writeVec4(view, 48, camera.horizontal);
  writeVec4(view, 64, camera.vertical);

  view.setUint32(80, depth, true);
  view.setUint32(84, samples, true);
  view.setUint32(88, frame, true);
  view.setUint32(92, 0, true);

  return buffer;
}

function writeVec4(view, offset, vec) {
  view.setFloat32(offset + 0, vec[0], true);
  view.setFloat32(offset + 4, vec[1], true);
  view.setFloat32(offset + 8, vec[2], true);
  view.setFloat32(offset + 12, vec[3], true);
}

function computeSuzanneCenter(scene) {
  const triStride = 48;
  const view = new DataView(scene.buffer, scene.byteOffset, scene.byteLength);
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let count = 0;

  for (let tri = 0; tri + triStride <= scene.byteLength; tri += triStride) {
    const material = view.getFloat32(tri + 12, true);
    if (Math.abs(material - 4.0) > 1e-6) {
      continue;
    }

    for (const base of [tri, tri + 16, tri + 32]) {
      sx += view.getFloat32(base + 0, true);
      sy += view.getFloat32(base + 4, true);
      sz += view.getFloat32(base + 8, true);
      count += 1;
    }
  }

  if (count === 0) {
    return [0.0, 0.0, -2.5];
  }

  return [sx / count, sy / count, sz / count];
}

function orbitCamera(target, timeSeconds, aspect) {
  const radius = 3.5;
  const height = 0.4;
  const speed = 0.7;
  const angle = timeSeconds * speed;

  const origin = [
    target[0] + radius * Math.cos(angle),
    target[1] + height,
    target[2] + radius * Math.sin(angle),
  ];

  return lookAt(origin, target, [0, 1, 0], 60.0, aspect);
}

function lookAt(origin, target, up, vfovDegrees, aspect) {
  const theta = (vfovDegrees * Math.PI) / 180.0;
  const h = Math.tan(theta * 0.5);
  const viewportHeight = 2.0 * h;
  const viewportWidth = aspect * viewportHeight;

  const w = normalize(sub(origin, target));
  const u = normalize(cross(up, w));
  const v = cross(w, u);

  const horizontal = scale(u, viewportWidth);
  const vertical = scale(v, viewportHeight);
  const lowerLeftCorner = sub(sub(sub(origin, scale(horizontal, 0.5)), scale(vertical, 0.5)), w);

  return {
    origin: [origin[0], origin[1], origin[2], 0.0],
    lowerLeftCorner: [lowerLeftCorner[0], lowerLeftCorner[1], lowerLeftCorner[2], 0.0],
    horizontal: [horizontal[0], horizontal[1], horizontal[2], 0.0],
    vertical: [vertical[0], vertical[1], vertical[2], 0.0],
  };
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v) {
  const len2 = dot(v, v);
  if (len2 <= 1e-12) {
    return [0, 0, 0];
  }
  const invLen = 1.0 / Math.sqrt(len2);
  return [v[0] * invLen, v[1] * invLen, v[2] * invLen];
}
