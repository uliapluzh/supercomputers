#include "redistribute.h"
#include "logging.h"

#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>   // getenv, atoi
#include <iostream>
#include <cstdint>

static uint64_t stableHash(const std::string &s)
{
    uint64_t h = 1469598103934665603ull; // FNV-1a
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}


static bool isGpuNode()
{
    const char* node = std::getenv("SLURMD_NODENAME");
    return node && std::strncmp(node, "gpu", 3) == 0;
}

// объявление (используется у тебя в compute.cpp)
extern bool canUseCUDA();

// ============================================================================
// Веса
// ============================================================================

static int getLocalWeight()
{
    const char* cpu = std::getenv("CPU_WEIGHT");
    const char* gpu = std::getenv("GPU_WEIGHT");

    int cpu_w = cpu ? std::atoi(cpu) : 1;
    int gpu_w = gpu ? std::atoi(gpu) : 3;

    return isGpuNode() ? gpu_w : cpu_w;
}


// ============================================================================
// Взвешенный ownerRank
// ============================================================================

static int ownerRankWeighted(
    const std::string &key,
    const std::vector<int> &prefix,
    int totalWeight
)
{
    uint64_t h = stableHash(key);
    int slot = static_cast<int>(h % totalWeight);

    auto it = std::upper_bound(prefix.begin(), prefix.end(), slot);
    return static_cast<int>(it - prefix.begin()) - 1;
}


// ============================================================================
// Основная функция redistribute
// ============================================================================

DataVec redistributeByKey(const DataVec &local)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);
    std::string hostname(host);

    double t0 = MPI_Wtime();

    // ------------------------------------------------------------------------
    // 1. Собираем веса всех rank'ов
    // ------------------------------------------------------------------------

    int myWeight = getLocalWeight();

    std::vector<int> allWeights(size);
    MPI_Allgather(
        &myWeight, 1, MPI_INT,
        allWeights.data(), 1, MPI_INT,
        MPI_COMM_WORLD
    );

    // prefix sum
    std::vector<int> prefix(size + 1);
    prefix[0] = 0;
    for (int i = 0; i < size; ++i)
        prefix[i + 1] = prefix[i] + allWeights[i];

    int totalWeight = prefix[size];

    // ------------------------------------------------------------------------
    // 2. Формируем send buffers
    // ------------------------------------------------------------------------

    std::vector<std::ostringstream> sendBuf(size);

    for (const auto &r : local) {
        int dst = ownerRankWeighted(r.key, prefix, totalWeight);
        sendBuf[dst] << r.key << ';'
                     << r.year << ';'
                     << r.temp << '\n';
    }

    // ------------------------------------------------------------------------
    // 3. Размеры сообщений
    // ------------------------------------------------------------------------

    std::vector<std::string> sendStr(size);
    std::vector<int> sendSizes(size);

    for (int i = 0; i < size; ++i) {
        sendStr[i]   = sendBuf[i].str();
        sendSizes[i] = static_cast<int>(sendStr[i].size());
    }

    std::vector<int> recvSizes(size);
    MPI_Alltoall(
        sendSizes.data(), 1, MPI_INT,
        recvSizes.data(), 1, MPI_INT,
        MPI_COMM_WORLD
    );

    // ------------------------------------------------------------------------
    // 4. Смещения
    // ------------------------------------------------------------------------

    std::vector<int> sdispls(size), rdispls(size);
    int stotal = 0, rtotal = 0;

    for (int i = 0; i < size; ++i) {
        sdispls[i] = stotal;
        rdispls[i] = rtotal;
        stotal += sendSizes[i];
        rtotal += recvSizes[i];
    }

    // ------------------------------------------------------------------------
    // 5. Alltoallv
    // ------------------------------------------------------------------------

    std::vector<char> sendBufFlat(stotal);
    std::vector<char> recvBufFlat(rtotal);

    for (int i = 0; i < size; ++i) {
        std::copy(
            sendStr[i].begin(),
            sendStr[i].end(),
            sendBufFlat.begin() + sdispls[i]
        );
    }

    MPI_Alltoallv(
        sendBufFlat.data(), sendSizes.data(), sdispls.data(), MPI_CHAR,
        recvBufFlat.data(), recvSizes.data(), rdispls.data(), MPI_CHAR,
        MPI_COMM_WORLD
    );

    // ------------------------------------------------------------------------
    // 6. Парсим результат
    // ------------------------------------------------------------------------

    DataVec result;
    result.reserve(local.size()); // эвристика

    std::string all(recvBufFlat.begin(), recvBufFlat.end());
    std::istringstream in(all);
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        Record r;
        std::string year, temp;

        std::getline(ss, r.key, ';');
        std::getline(ss, year,  ';');
        std::getline(ss, temp,  ';');

        r.year = std::stoi(year);
        r.temp = std::stod(temp);

        result.push_back(r);
    }

    double t1 = MPI_Wtime();
    log_event(rank, hostname, size, "redistribute", t0, t1);

    return result;
}
