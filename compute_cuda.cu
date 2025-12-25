#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

// ======================================================
// CUDA KERNEL
// ======================================================
__global__ void compute_min_delta_kernel(
    const double* values,
    const int* offsets,
    const int* lengths,
    double* out_min_delta
) {
    int series_id = blockIdx.x;
    int tid = threadIdx.x;

    int offset = offsets[series_id];
    int len    = lengths[series_id];

    extern __shared__ double shmem[];

    double local_min = 1e300;

    for (int i = tid + 1; i < len; i += blockDim.x) {
        double d = fabs(values[offset + i] - values[offset + i - 1]);
        if (d < local_min) local_min = d;
    }

    shmem[tid] = local_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shmem[tid + s] < shmem[tid])
            shmem[tid] = shmem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        out_min_delta[series_id] = (len < 2) ? 1e300 : shmem[0];
}

// ======================================================
// HOST API
// ======================================================
extern "C" bool compute_min_delta_cuda(
    const std::vector<double>& values,
    const std::vector<int>& offsets,
    const std::vector<int>& lengths,
    std::vector<double>& out_min_delta
) {
    // ⚠️ ВСЕ объявления — В САМОМ НАЧАЛЕ
    double* d_values  = nullptr;
    int*    d_offsets = nullptr;
    int*    d_lengths = nullptr;
    double* d_out     = nullptr;

    cudaError_t err = cudaSuccess;

    int blockSize = 256;
    int gridSize  = 0;
    size_t sharedMem = 0;

    size_t valuesBytes  = 0;
    size_t offsetsBytes = 0;
    size_t lengthsBytes = 0;
    size_t outBytes     = 0;

    // CUDA-ветка НИКОГДА не выпадает
    if (offsets.empty()) {
        out_min_delta.clear();
        return true;
    }

    valuesBytes  = values.size()  * sizeof(double);
    offsetsBytes = offsets.size() * sizeof(int);
    lengthsBytes = lengths.size() * sizeof(int);
    outBytes     = offsets.size() * sizeof(double);

    // ---- allocations ----
    if ((err = cudaMalloc(&d_values, valuesBytes))  != cudaSuccess) goto fail;
    if ((err = cudaMalloc(&d_offsets, offsetsBytes))!= cudaSuccess) goto fail;
    if ((err = cudaMalloc(&d_lengths, lengthsBytes))!= cudaSuccess) goto fail;
    if ((err = cudaMalloc(&d_out, outBytes))        != cudaSuccess) goto fail;

    // ---- copies ----
    cudaMemcpy(d_values,  values.data(),  valuesBytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsetsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), lengthsBytes, cudaMemcpyHostToDevice);

    gridSize  = offsets.size();
    sharedMem = blockSize * sizeof(double);

    compute_min_delta_kernel<<<gridSize, blockSize, sharedMem>>>(
        d_values, d_offsets, d_lengths, d_out
    );

    if ((err = cudaGetLastError()) != cudaSuccess) goto fail;

    out_min_delta.resize(offsets.size());
    cudaMemcpy(out_min_delta.data(), d_out, outBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_out);
    return true;

fail:
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (d_values)  cudaFree(d_values);
    if (d_offsets) cudaFree(d_offsets);
    if (d_lengths) cudaFree(d_lengths);
    if (d_out)     cudaFree(d_out);
    return false;
}

// ======================================================
// dlopen API
// ======================================================
extern "C"
std::vector<double>
computeMinDeltasCUDA(
    const std::vector<double>& values,
    const std::vector<int>& offsets,
    const std::vector<int>& lengths
) {
    std::vector<double> out;
    if (!compute_min_delta_cuda(values, offsets, lengths, out))
        return {};
    return out;
}
