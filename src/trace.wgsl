

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

struct Params {
    width: u32,
    height: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

// learnt that if you dont use the buffer in the code it wont be generate and you will get an error

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    let d = input[0];

    // Guard against extra threads
    if (x >= params.width || y >= params.height) {
        return;
    }

    // 1D index into output buffer
    let index = y * params.width + x;

    // Normalized coordinates [0, 1]
    let fx = f32(x) / f32(params.width - 1u);
    let fy = f32(y) / f32(params.height - 1u);

    // Simple gradient
    let r: u32 = u32(fx * 255.0);
    let g: u32 = u32(fy * 255.0);
    let b: u32 = 128u;

    // Pack as 0x00RRGGBB
    output[index] = (r << 16u) | (g << 8u) | b;
}
