#include "reduce.h"
#include "logging.h"

#include <mpi.h>
#include <sstream>
#include <algorithm>

std::vector<MinDelta>
reduceMinDeltasMPI(const std::vector<MinDelta> &local)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);
    std::string hostname(host);

     //MPI_Barrier(MPI_COMM_WORLD);
    //  double t0 = MPI_Wtime();

    /* --- serialize local --- */
    std::ostringstream oss;
    for (const auto &md : local)
        oss << md.key << ";" << md.delta << "\n";

    std::string sendStr = oss.str();
    int sendSize = sendStr.size();

    /* --- gather sizes --- */
    std::vector<int> recvSizes;
    if (rank == 0)
        recvSizes.resize(size);

    MPI_Gather(&sendSize, 1, MPI_INT,
               recvSizes.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    /* --- gather buffers --- */
    std::vector<int> displs;
    std::vector<char> recvBuf;
    int total = 0;

    if (rank == 0) {
        displs.resize(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += recvSizes[i];
        }
        recvBuf.resize(total);
    }

    MPI_Gatherv(sendStr.data(), sendSize, MPI_CHAR,
                recvBuf.data(), recvSizes.data(), displs.data(),
                MPI_CHAR, 0, MPI_COMM_WORLD);

    /* --- parse + sort on rank 0 --- */
    std::vector<MinDelta> global;

    if (rank == 0) {
        std::string all(recvBuf.begin(), recvBuf.end());
        std::istringstream in(all);
        std::string line;

        while (std::getline(in, line)) {
            std::stringstream ss(line);
            MinDelta md;
            std::string d;

            std::getline(ss, md.key, ';');
            std::getline(ss, d, ';');
            md.delta = std::stod(d);

            global.push_back(md);
        }

        std::sort(global.begin(), global.end(),
                  [](auto &a, auto &b) {
                      return a.delta < b.delta;
                  });
    }

    //  MPI_Barrier(MPI_COMM_WORLD);
    //  double t1 = MPI_Wtime();
    //  log_event(rank, hostname, size, "reduce_min_delta", t0, t1);

    return global;  
}
