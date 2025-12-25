#include "compute.h"
#include "logging.h"

#include <mpi.h>
#include <dlfcn.h>

#include <algorithm>
#include <limits>
#include <iostream>
#include <cstdlib>   // getenv
#include <cstring>   // strcmp, strlen

// ================= CUDA plugin interface =================

using CudaFn = std::vector<double>(*)(
    const std::vector<double>&,
    const std::vector<int>&,
    const std::vector<int>&
);

bool canUseCUDA()
{
    const char* use_cuda = std::getenv("USE_CUDA");
    if (!use_cuda || std::strcmp(use_cuda, "1") != 0)
        return false;

    const char* node = std::getenv("SLURMD_NODENAME");
    if (!node)
        return false;

    // GPU-ноды
    if (std::strncmp(node, "gpu", 3) == 0)
        return true;

    return false;
}

CudaFn loadCudaFunction()
{
    if (!canUseCUDA())
        return nullptr;

    void* handle = dlopen("./libmin_delta_cuda.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << std::endl;
        return nullptr;
    }

    auto fn = reinterpret_cast<CudaFn>(
        dlsym(handle, "computeMinDeltasCUDA")
    );

    if (!fn) {
        std::cerr << "dlsym failed: " << dlerror() << std::endl;
        return nullptr;
    }

    return fn;
}

// ================= Core computations =================

PartialMap computeLocalPartials(const DataVec &data)
{
    PartialMap p;
    for (const auto &r : data) {
        auto &s = p[r.key][r.year];
        s.sum   += r.temp;
        s.count += 1;
    }
    return p;
}

YearlyAverages computeFinalAverages(const PartialMap &p)
{
    YearlyAverages y;
    for (const auto &[key, years] : p)
        for (const auto &[year, stat] : years)
            y[key][year] = stat.sum / stat.count;
    return y;
}

std::vector<MinDelta>
computeMinDeltas(const YearlyAverages &yearly)
{
    std::vector<double> values;
    std::vector<int> offsets;
    std::vector<int> lengths;
    std::vector<std::string> keys;

    int offset = 0;

    // --- flatten ---
    for (const auto &[key, ym] : yearly) {
        if (ym.size() < 2) continue;

        offsets.push_back(offset);
        lengths.push_back(static_cast<int>(ym.size()));
        keys.push_back(key);

        for (const auto &[year, temp] : ym) {
            values.push_back(temp);
            offset++;
        }
    }

    std::vector<double> deltas;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(host, &len);

    std::cerr
        << "[rank " << rank << " | " << host << "] "
        << "series=" << keys.size()
        << " values=" << values.size()
        << std::endl;

    // --- try CUDA ---
    if (auto cudaFn = loadCudaFunction()) {
        std::cerr << "[rank " << rank << "] CUDA ENABLED\n";
        deltas = cudaFn(values, offsets, lengths);
    } else {
        std::cerr << "[rank " << rank << "] CPU fallback\n";

        for (size_t i = 0; i < offsets.size(); ++i) {
            int start = offsets[i];
            int len   = lengths[i];

            double best = std::numeric_limits<double>::max();
            for (int j = 1; j < len; ++j) {
                double d = std::abs(values[start + j]
                                  - values[start + j - 1]);
                best = std::min(best, d);
            }
            deltas.push_back(best);
        }
    }

    // --- pack result ---
    std::vector<MinDelta> res;
    for (size_t i = 0; i < deltas.size(); ++i)
        res.push_back({ keys[i], deltas[i] });

    std::sort(res.begin(), res.end(),
              [](const MinDelta &a, const MinDelta &b) {
                  return a.delta < b.delta;
              });

    return res;
}

// ================= CPU-only helpers =================

std::vector<FlatSeries>
flattenYearlyAverages(const YearlyAverages &yearly)
{
    std::vector<FlatSeries> out;

    for (const auto &[key, ym] : yearly) {
        if (ym.size() < 2) continue;

        FlatSeries fs;
        fs.key = key;

        for (const auto &[year, val] : ym)
            fs.values.push_back(val);

        out.push_back(std::move(fs));
    }

    return out;
}

std::vector<MinDelta>
computeMinDeltasCPU(const std::vector<FlatSeries> &series)
{
    std::vector<MinDelta> res;

    for (const auto &fs : series) {
        const auto &v = fs.values;
        if (v.size() < 2) continue;

        double best = std::numeric_limits<double>::max();
        for (size_t i = 1; i < v.size(); ++i)
            best = std::min(best, std::abs(v[i] - v[i - 1]));

        res.push_back({ fs.key, best });
    }

    std::sort(res.begin(), res.end(),
              [](const MinDelta &a, const MinDelta &b) {
                  return a.delta < b.delta;
              });

    return res;
}
