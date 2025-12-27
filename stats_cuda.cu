#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <cmath>
#include <vector>

struct GpuMinDelta {
    int key_id;
    double min_delta;
};

// -------- helpers --------

struct SumCount {
    double sum;
    int count;
};

struct SumCountPlus {
    __host__ __device__
    SumCount operator()(const SumCount &a,
                        const SumCount &b) const {
        return { a.sum + b.sum, a.count + b.count };
    }
};

struct MakeSumCount {
    __host__ __device__
    SumCount operator()(double t) const {
        return { t, 1 };
    }
};

struct ComputeAvg {
    __host__ __device__
    double operator()(const SumCount &sc) const {
        return sc.sum / sc.count;
    }
};

// ===== ВАЖНОЕ ИСПРАВЛЕНИЕ =====

struct DeltaFunctor {
    const double* avg;
    const int*    keys;

    __host__ __device__
    double operator()(int i) const {
        if (i == 0) return 1e300;
        if (keys[i] != keys[i - 1]) return 1e300;
        return fabs(avg[i] - avg[i - 1]);
    }
};

// -------- main CUDA function --------
// ВАЖНО: вход уже отсортирован по (key, year) на CPU

extern "C"
std::vector<GpuMinDelta>
computeStatsCUDA(const std::vector<int> &keys,
                 const std::vector<int> &years,
                 const std::vector<double> &temps,
                 int num_keys)
{
    int n = keys.size();

    thrust::device_vector<int>    d_keys(keys.begin(), keys.end());
    thrust::device_vector<int>    d_years(years.begin(), years.end());
    thrust::device_vector<double> d_temps(temps.begin(), temps.end());

    auto zip_key_year =
        thrust::make_zip_iterator(
            thrust::make_tuple(d_keys.begin(), d_years.begin())
        );

    thrust::device_vector<SumCount> d_sc(n);
    thrust::transform(
        d_temps.begin(), d_temps.end(),
        d_sc.begin(),
        MakeSumCount{}
    );

    thrust::device_vector<int>      d_keys2(n);
    thrust::device_vector<int>      d_years2(n);
    thrust::device_vector<SumCount> d_sc2(n);

    auto new_end = thrust::reduce_by_key(
        zip_key_year,
        zip_key_year + n,
        d_sc.begin(),
        thrust::make_zip_iterator(
            thrust::make_tuple(d_keys2.begin(), d_years2.begin())
        ),
        d_sc2.begin(),
        thrust::equal_to<thrust::tuple<int,int>>{},
        SumCountPlus{}
    );

    int m = new_end.second - d_sc2.begin();

    thrust::device_vector<double> d_avg(m);
    thrust::transform(
        d_sc2.begin(), d_sc2.begin() + m,
        d_avg.begin(),
        ComputeAvg{}
    );

    // ===== БЕЗ UB, БЕЗ device-lambda =====

    thrust::device_vector<double> d_delta(m);

    DeltaFunctor fn {
        d_avg.data().get(),
        d_keys2.data().get()
    };

    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(m),
        d_delta.begin(),
        fn
    );

    thrust::device_vector<int>    d_out_keys(num_keys);
    thrust::device_vector<double> d_out_vals(num_keys);

    auto out_end = thrust::reduce_by_key(
        d_keys2.begin(),
        d_keys2.begin() + m,
        d_delta.begin(),
        d_out_keys.begin(),
        d_out_vals.begin(),
        thrust::equal_to<int>{},
        thrust::minimum<double>{}
    );

    int r = out_end.second - d_out_vals.begin();

    thrust::host_vector<int>    h_keys(d_out_keys.begin(), d_out_keys.begin() + r);
    thrust::host_vector<double> h_vals(d_out_vals.begin(), d_out_vals.begin() + r);

    std::vector<GpuMinDelta> out;
    out.reserve(r);
    for (int i = 0; i < r; ++i)
        out.push_back({ h_keys[i], h_vals[i] });

    return out;
}
