#include "redistribute.h"
#include "logging.h"

#include <mpi.h>
#include <vector>
#include <sstream>

static int ownerRank(const std::string &key, int size)
{
    return std::hash<std::string>{}(key) % size;
}

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

    std::vector<std::ostringstream> sendBuf(size);

    for (const auto &r : local) {
        int dst = ownerRank(r.key, size);
        sendBuf[dst] << r.key << ';'
                     << r.year << ';'
                     << r.temp << '\n';
    }

    std::vector<std::string> sendStr(size);
    std::vector<int> sendSizes(size);

    for (int i = 0; i < size; ++i) {
        sendStr[i] = sendBuf[i].str();
        sendSizes[i] = sendStr[i].size();
    }

    std::vector<int> recvSizes(size);
    MPI_Alltoall(sendSizes.data(), 1, MPI_INT,
                 recvSizes.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> sdispls(size), rdispls(size);
    int stotal = 0, rtotal = 0;

    for (int i = 0; i < size; ++i) {
        sdispls[i] = stotal;
        rdispls[i] = rtotal;
        stotal += sendSizes[i];
        rtotal += recvSizes[i];
    }

    std::vector<char> sendBufFlat(stotal);
    std::vector<char> recvBufFlat(rtotal);

    for (int i = 0; i < size; ++i)
        std::copy(sendStr[i].begin(), sendStr[i].end(),
                  sendBufFlat.begin() + sdispls[i]);

    MPI_Alltoallv(sendBufFlat.data(), sendSizes.data(), sdispls.data(), MPI_CHAR,
                  recvBufFlat.data(), recvSizes.data(), rdispls.data(), MPI_CHAR,
                  MPI_COMM_WORLD);

    DataVec result;
    std::string all(recvBufFlat.begin(), recvBufFlat.end());
    std::istringstream in(all);
    std::string line;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        Record r;
        std::string year, temp;

        std::getline(ss, r.key, ';');
        std::getline(ss, year, ';');
        std::getline(ss, temp, ';');

        r.year = std::stoi(year);
        r.temp = std::stod(temp);

        result.push_back(r);
    }

    double t1 = MPI_Wtime();
    log_event(rank, hostname, size, "redistribute", t0, t1);

    return result;
}
