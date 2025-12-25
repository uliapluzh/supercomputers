#include "logging.h"
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <vector>

void log_event(int rank,
               const std::string &host,
               int size,
               const std::string &op,
               double t_start,
               double t_end)
{
    std::ostringstream oss;
    oss << rank << ","
        << host << ","
        << size << ","
        << op << ","
        << t_start << ","
        << t_end << ","
        << (t_end - t_start) << "\n";

    std::string line = oss.str();
    int len = line.size();

    std::vector<int> sizes;
    if (rank == 0)
        sizes.resize(size);

    MPI_Gather(&len, 1, MPI_INT,
               sizes.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs;
    std::vector<char> buffer;

    if (rank == 0) {
        displs.resize(size);
        int total = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += sizes[i];
        }
        buffer.resize(total);
    }

    MPI_Gatherv(line.data(), len, MPI_CHAR,
                buffer.data(), sizes.data(), displs.data(),
                MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream out("timeline.csv", std::ios::app);
        out.write(buffer.data(), buffer.size());
    }
}
