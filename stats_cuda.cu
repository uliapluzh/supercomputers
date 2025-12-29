#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>

// ================= STRUCT =================

struct GpuMinDelta {
    int    key_id;
    double min_delta;
};

constexpr int BLOCK_SIZE = 128;

// ================= CUDA KERNEL =================

__global__
void min_delta_from_avg_kernel(
    const double* avg,
    const int*    key_offsets,
    const int*    key_sizes,
    int           num_keys,
    double*       out
)
{
    int k = blockIdx.x;
    if (k >= num_keys) return;

    int start = key_offsets[k];
    int len   = key_sizes[k];

    if (len < 2) {
        if (threadIdx.x == 0)
            out[k] = 0.0;
        return;
    }

    __shared__ double sh_min[BLOCK_SIZE];

    double local_min = 1e300;

    // |avg[i+1] - avg[i]|
    for (int i = threadIdx.x; i < len - 1; i += blockDim.x) {
        double d = fabs(avg[start + i + 1] - avg[start + i]);
        local_min = fmin(local_min, d);
    }

    sh_min[threadIdx.x] = local_min;
    __syncthreads();

    // reduction min
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_min[threadIdx.x] =
                fmin(sh_min[threadIdx.x], sh_min[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[k] = sh_min[0];
}

// ================= HOST INTERFACE =================

extern "C"
std::vector<GpuMinDelta>
computeStatsCUDA(
    const std::vector<double> &avg,
    const std::vector<int>    &key_offsets,
    const std::vector<int>    &key_sizes,
    int num_keys
)
{
    int n = avg.size();

    // ===== CUDA BUFFER CACHE =====
    static double *d_avg     = nullptr;
    static double *d_out     = nullptr;
    static int    *d_offsets = nullptr;
    static int    *d_sizes   = nullptr;

    static int cap_avg  = 0;
    static int cap_keys = 0;

    // avg buffer
    if (n > cap_avg) {
        if (d_avg) cudaFree(d_avg);
        cudaMalloc(&d_avg, n * sizeof(double));
        cap_avg = n;
    }

    // offsets / sizes / output
    if (num_keys > cap_keys) {
        if (d_offsets) cudaFree(d_offsets);
        if (d_sizes)   cudaFree(d_sizes);
        if (d_out)     cudaFree(d_out);

        cudaMalloc(&d_offsets, num_keys * sizeof(int));
        cudaMalloc(&d_sizes,   num_keys * sizeof(int));
        cudaMalloc(&d_out,     num_keys * sizeof(double));

        cap_keys = num_keys;
    }

    // ===== H2D =====
    cudaMemcpy(d_avg,
               avg.data(),
               n * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_offsets,
               key_offsets.data(),
               num_keys * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_sizes,
               key_sizes.data(),
               num_keys * sizeof(int),
               cudaMemcpyHostToDevice);

    // ===== KERNEL =====
    min_delta_from_avg_kernel<<<num_keys, BLOCK_SIZE>>>(
        d_avg,
        d_offsets,
        d_sizes,
        num_keys,
        d_out
    );

    // cudaMemcpy D2H уже синхронизирует устройство
    std::vector<double> h_out(num_keys);
    cudaMemcpy(h_out.data(),
               d_out,
               num_keys * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ===== RESULT =====
    std::vector<GpuMinDelta> res;
    res.reserve(num_keys);

    for (int i = 0; i < num_keys; ++i) {
        res.push_back({ i, h_out[i] });
    }

    return res;
}
