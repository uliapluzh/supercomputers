#include <mpi.h>
#include <fstream>
#include "reduce.h"
#include "reader.h"
#include "redistribute.h"
#include "compute.h"
#include "logging.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);
    std::string hostname(host);

    if (rank == 0) {
        std::ofstream out("timeline.csv");
        out << "rank,host,size,operation,t_start,t_end,duration\n";
    }

    double t_prog = MPI_Wtime();

    DataVec local = readCSVChunk("GlobalLandTemperaturesByCity.csv");
    DataVec owned = redistributeByKey(local);

    // ----------------- MIN DELTA -----------------
    double t0 = MPI_Wtime();
    auto localDeltas = computeLocalStats(owned);
    double t1 = MPI_Wtime();
    log_event(rank, hostname, size, "final_compute", t0, t1);

    // ----------------- REDUCE -----------------
    double t2 = MPI_Wtime();
    auto deltas = reduceMinDeltasMPI(localDeltas);
    double t3 = MPI_Wtime();
    log_event(rank, hostname, size, "reduce_min_delta", t2, t3);

    // // ----------------- YEARLY (rank 0 only for debug output) -----------------
    // YearlyAverages yearly;
    // if (rank == 0) {
    //     auto partials = computeLocalPartials(owned);
    //     yearly = computeFinalAverages(partials);
    // }

    if (rank == 0) {
    std::cerr << "[DEBUG] global min_deltas = "
              << deltas.size() << std::endl;
    }

    if (rank == 0) {
        // std::ofstream out("output.txt");
        // out << "Country,City,Year,AverageTemperature\n";

        // for (const auto &cp : yearly) {
        //     auto sep = cp.first.find('|');
        //     for (const auto &yp : cp.second)
        //         out << cp.first.substr(0, sep) << ","
        //             << cp.first.substr(sep + 1) << ","
        //             << yp.first << ","
        //             << yp.second << "\n";
        // }

        std::ofstream out2("min_delta.txt");
        out2 << "Country,City,MinAbsYearlyDelta\n";

        int count = 0;
        for (const auto &md : deltas) {
            if (count >= 100) break;
            if (md.delta <= 0.0001) continue;

            auto sep = md.key.find('|');
            out2 << md.key.substr(0, sep) << ","
                 << md.key.substr(sep + 1) << ","
                 << md.delta << "\n";
            ++count;
        }
    }

    /* ВСЕ ранки логируют program_total */
    log_event(rank, hostname, size,
              "program_total", t_prog, MPI_Wtime());

    MPI_Finalize();
    return 0;
}