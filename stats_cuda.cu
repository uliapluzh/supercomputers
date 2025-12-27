#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

struct GpuMinDelta {
    int    key_id;
    double min_delta;
};

// ================== CUDA kernel ==================

__global__
void compute_min_delta_kernel(
    const int*    keys,
    const int*    years,
    const double* temps,
    const int*    key_offsets,   // начало каждого key
    const int*    key_sizes,     // длина каждого key
    int           num_keys,
    double*       out_min_delta
)
{
    int k = blockIdx.x;
    if (k >= num_keys) return;

    int start = key_offsets[k];
    int len   = key_sizes[k];

    double prev_avg = 0.0;
    double min_d    = 1e300;

    int i = 0;
    while (i < len) {
        int year = years[start + i];
        double sum = 0.0;
        int cnt = 0;

        // агрегируем один год
        while (i < len && years[start + i] == year) {
            sum += temps[start + i];
            cnt++;
            i++;
        }

        double avg = sum / cnt;

        if (cnt > 0 && prev_avg != 0.0) {
            double d = fabs(avg - prev_avg);
            if (d < min_d) min_d = d;
        }

        prev_avg = avg;
    }

    out_min_delta[k] = min_d;
}

// ================== HOST interface ==================

extern "C"
std::vector<GpuMinDelta>
computeStatsCUDA(
    const std::vector<int>    &keys,
    const std::vector<int>    &years,
    const std::vector<double> &temps,
    int num_keys
)
{
    int n = keys.size();

    // ---------- вычисляем offsets на CPU ----------
    std::vector<int> key_offsets(num_keys);
    std::vector<int> key_sizes(num_keys);

    int cur_key = -1;
    for (int i = 0; i < n; ++i) {
        if (i == 0 || keys[i] != keys[i - 1]) {
            cur_key++;
            key_offsets[cur_key] = i;
            key_sizes[cur_key] = 1;
        } else {
            key_sizes[cur_key]++;
        }
    }

    // ---------- device memory ----------
    int    *d_keys, *d_years, *d_key_offsets, *d_key_sizes;
    double *d_temps, *d_out;

    cudaMalloc(&d_keys,        n * sizeof(int));
    cudaMalloc(&d_years,       n * sizeof(int));
    cudaMalloc(&d_temps,       n * sizeof(double));
    cudaMalloc(&d_key_offsets,num_keys * sizeof(int));
    cudaMalloc(&d_key_sizes,  num_keys * sizeof(int));
    cudaMalloc(&d_out,        num_keys * sizeof(double));

    cudaMemcpy(d_keys, keys.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_years, years.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temps, temps.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_offsets, key_offsets.data(),
               num_keys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_sizes, key_sizes.data(),
               num_keys * sizeof(int), cudaMemcpyHostToDevice);

    // ---------- kernel ----------
    compute_min_delta_kernel<<<num_keys, 1>>>(
        d_keys,
        d_years,
        d_temps,
        d_key_offsets,
        d_key_sizes,
        num_keys,
        d_out
    );

    cudaDeviceSynchronize();

    // ---------- copy back ----------
    std::vector<double> h_out(num_keys);
    cudaMemcpy(h_out.data(), d_out,
               num_keys * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ---------- cleanup ----------
    cudaFree(d_keys);
    cudaFree(d_years);
    cudaFree(d_temps);
    cudaFree(d_key_offsets);
    cudaFree(d_key_sizes);
    cudaFree(d_out);

    // ---------- pack result ----------
    std::vector<GpuMinDelta> result;
    result.reserve(num_keys);

    for (int i = 0; i < num_keys; ++i) {
        result.push_back({ i, h_out[i] });
    }

    return result;
}
