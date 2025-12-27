#include "compute.h"

#include <mpi.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <limits>

// ================= CUDA plugin interface =================

struct GpuMinDelta {
    int key_id;
    double min_delta;
};

using CudaStatsFn =
std::vector<GpuMinDelta>(*)(
    const std::vector<int>&,
    const std::vector<int>&,
    const std::vector<double>&,
    int
);

static CudaStatsFn loadCudaStatsFunction()
{
    void* h = dlopen("./libstats_cuda.so", RTLD_LAZY);
    if (!h)
        return nullptr;

    return reinterpret_cast<CudaStatsFn>(
        dlsym(h, "computeStatsCUDA")
    );
}

bool canUseCUDA()
{
    const char* use = std::getenv("USE_CUDA");
    if (!use || std::strcmp(use, "1") != 0)
        return false;

    const char* node = std::getenv("SLURMD_NODENAME");
    return node && std::strncmp(node, "gpu", 3) == 0;
}

// ================= CPU implementation =================

static std::vector<MinDelta>
computeStatsCPU(const DataVec &data)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);

    // key → year → (sum, count)
    std::unordered_map<std::string,
        std::map<int, std::pair<double,int>>> acc;

    for (const auto &r : data) {
        auto &cell = acc[r.key][r.year];
        cell.first  += r.temp;
        cell.second += 1;
    }

    std::cerr
        << "[rank " << rank << " | " << host << "] "
        << "series=" << acc.size()
        << " values=" << data.size()
        << " (CPU)"
        << std::endl;

    std::vector<MinDelta> res;

    for (const auto &[key, years] : acc) {
        if (years.size() < 2)
            continue;

        double best = std::numeric_limits<double>::max();

        auto it = years.begin();
        double prev = it->second.first / it->second.second;
        ++it;

        for (; it != years.end(); ++it) {
            double cur = it->second.first / it->second.second;
            best = std::min(best, std::abs(cur - prev));
            prev = cur;
        }

        res.push_back({ key, best });
    }

    std::sort(res.begin(), res.end(),
              [](const MinDelta &a, const MinDelta &b) {
                  return a.delta < b.delta;
              });

    return res;
}

// ================= GPU wrapper =================

static std::vector<MinDelta>
computeStatsCUDA_wrapper(const DataVec &data)
{
    std::unordered_map<std::string,int> key2id;
    std::vector<std::string> id2key;

    std::vector<int>    keys;
    std::vector<int>    years;
    std::vector<double> temps;

    for (const auto &r : data) {
        int id;
        auto it = key2id.find(r.key);
        if (it == key2id.end()) {
            id = key2id.size();
            key2id[r.key] = id;
            id2key.push_back(r.key);
        } else {
            id = it->second;
        }

        keys.push_back(id);
        years.push_back(r.year);
        temps.push_back(r.temp);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);

    std::cerr
        << "[rank " << rank << " | " << host << "] "
        << "series=" << id2key.size()
        << " values=" << keys.size()
        << " (GPU)"
        << std::endl;

    auto fn = loadCudaStatsFunction();
    if (!fn) {
        std::cerr
            << "[rank " << rank
            << "] CUDA plugin not available, fallback to CPU\n";
        return computeStatsCPU(data);
    }

    auto gpuRes = fn(keys, years, temps, id2key.size());

    std::vector<MinDelta> res;
    res.reserve(gpuRes.size());

    for (const auto &g : gpuRes)
        res.push_back({ id2key[g.key_id], g.min_delta });

    std::sort(res.begin(), res.end(),
              [](const MinDelta &a, const MinDelta &b) {
                  return a.delta < b.delta;
              });

    return res;
}

// ================= Unified entry =================

std::vector<MinDelta>
computeLocalStats(const DataVec &input)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 🔥 КЛЮЧЕВОЕ МЕСТО
    // Явно сортируем ОДИН РАЗ: (key, year)
    DataVec data = input;
    std::sort(data.begin(), data.end(),
        [](const Record &a, const Record &b) {
            if (a.key != b.key)
                return a.key < b.key;
            return a.year < b.year;
        });

    if (canUseCUDA()) {
        std::cerr << "[rank " << rank << "] GPU path\n";
        return computeStatsCUDA_wrapper(data);
    }

    std::cerr << "[rank " << rank << "] CPU path\n";
    return computeStatsCPU(data);
}
