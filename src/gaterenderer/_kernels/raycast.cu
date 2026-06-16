#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct vec3 {
    float x, y, z;
    __device__ vec3() : x(0), y(0), z(0) {}
    __device__ vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    __device__ vec3 operator+(const vec3& o) const { return vec3(x+o.x, y+o.y, z+o.z); }
    __device__ vec3 operator-(const vec3& o) const { return vec3(x-o.x, y-o.y, z-o.z); }
    __device__ vec3 operator*(float s) const { return vec3(x*s, y*s, z*s); }
};

__device__ inline float dot(const vec3& a, const vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Per-quad geometry, precomputed once on the host (see GateRenderer.set_quads)
// and passed in as a global-memory buffer [n_quads, 16] so each renderer owns
// its own geometry (no shared __constant__ state). All threads in a warp read
// the same quad slot, so these loads broadcast through L1.
//   [0..2]  center of gate
//   [3..5]  normal = forward vector
//   [6..8]  u = left vector
//   [9..11] v = up vector
//   [12]    nd = dot(normal, center)        (plane offset)
//   [13]    inv_size = 1 / size
//   [14]    zero padding for 128-bit alignment
//   [15]    zero padding for 128-bit alignment

__global__ void raycast_quads_kernel(
    const float4* __restrict__ rays_cam, // Precomputed camera space rays [H, W, 4] (xyz used)
    const float* __restrict__ c2w,       // Camera to world matrices [n_cams, 4, 4]
    const float* __restrict__ quads_p,     // Packed per-quad geometry [n_quads, 16]
    uint8_t* __restrict__ output,
    int n_cams, int n_quads, int W, int H)
{
    // Stage the (warp-uniform) quad geometry into shared memory once per block.
    // Every thread reads the same quad slot in the loop below, so serving those
    // reads from shared recovers the constant-memory broadcast behaviour while
    // keeping the geometry a per-renderer global-memory argument.
    // NOTE: done before any early-return so all threads reach __syncthreads().
    extern __shared__ float s_quads_p[];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int n_floats = n_quads * 16;
    for (int i = tid; i < n_floats; i += blockDim.x * blockDim.y) {
        s_quads_p[i] = quads_p[i];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= W || y >= H || c >= n_cams) return;

    // Single coalesced 128-bit load of the precomputed camera-space ray.
    float4 r = rays_cam[y * W + x];
    vec3 ray_d_cam(r.x, r.y, r.z);

    // Fetch the Camera-To-World matrix for this camera
    const float* mat = &c2w[c * 16];

    // Ray origin is the translation component of the C2W matrix
    vec3 ray_o(mat[3], mat[7], mat[11]);

    // Rotate the ray direction to world space, NWU convention: x-forward, y-left, z-up
    vec3 ray_d_world(
        -mat[1]*ray_d_cam.x + mat[2]*ray_d_cam.y  - mat[0]*ray_d_cam.z,
        -mat[5]*ray_d_cam.x + mat[6]*ray_d_cam.y  - mat[4]*ray_d_cam.z,
        -mat[9]*ray_d_cam.x + mat[10]*ray_d_cam.y - mat[8]*ray_d_cam.z
    );

    bool hit = false;
#pragma unroll 1 // Unroll limit to avoid heavy register usage
    for(int q = 0; q < n_quads; ++q) {
        const float* qd = &s_quads_p[q * 16];

        vec3 normal(qd[3], qd[4], qd[5]);
        float denom = dot(ray_d_world, normal);
        if (fabsf(denom) < 1e-6f) continue;

        // t = dot(center - ray_o, normal) / denom, with dot(center, normal) precomputed as nd.
        float t = (qd[12] - dot(normal, ray_o)) / denom;
        if (t < 0.0f) continue;

        vec3 P = ray_o + ray_d_world * t;
        vec3 center(qd[0], qd[1], qd[2]);
        vec3 to_p = P - center;
        
        // Project the hit point onto the quad's local axes
        vec3 edge1(qd[6], qd[7], qd[8]);
        float u = dot(to_p, edge1) * qd[13];
        if (u < -1.0f || u > 1.0f) continue;

        vec3 edge2(qd[9], qd[10], qd[11]);
        float v = dot(to_p, edge2) * qd[13];
        if (fabsf(v) > 1.0f) continue;

        // Inside the quad: reject the central hole, otherwise it is a frame hit.
        if (fabsf(u) < 0.75f && fabsf(v) < 0.75f) continue;
        hit = true; // no break
    }

    output[c * W * H + y * W + x] = hit ? 1 : 0;
}

void launch_raycast_kernel(torch::Tensor rays_cam, torch::Tensor c2w, torch::Tensor quads_p, torch::Tensor output_buf, int W, int H) {
    int n_cams = c2w.size(0);
    int n_quads = quads_p.size(0);

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16, n_cams);
    size_t shmem = n_quads * 16 * sizeof(float);

    raycast_quads_kernel<<<blocks, threads, shmem>>>(
        reinterpret_cast<const float4*>(rays_cam.data_ptr<float>()),
        c2w.data_ptr<float>(),
        quads_p.data_ptr<float>(),
        output_buf.data_ptr<uint8_t>(),
        n_cams, n_quads, W, H
    );
}
