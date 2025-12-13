#include <iostream>
#include <fstream>
#include <mpi.h>

#include "reader.h"
#include "compute.h"

/* ---------- утилита логирования ---------- */
static void log_event(int rank,
                      const std::string &host,
                      int size,
                      const std::string &op,
                      double t0,
                      double t1)
{
    std::ofstream out("timeline.csv", std::ios::app);
    out << rank << ","
        << host << ","
        << size << ","
        << op << ","
        << t0 << ","
        << t1 << ","
        << (t1 - t0) << "\n";
}

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

    /* ---------- заголовок timeline ---------- */
    if (rank == 0) {
        std::ofstream out("timeline.csv");
        out << "rank,host,size,operation,t_start,t_end,duration\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t_program_start = MPI_Wtime();

    /* ---------- ЭТАП 1: чтение CSV (MPI) ---------- */
    DataMap data = readCSV("GlobalLandTemperaturesByCity.csv");

    if (rank == 0) {
        std::cout << "Read entries: " << data.size() << std::endl;

        /* ---------- ЭТАП 2: годовые средние ---------- */
        double t0 = MPI_Wtime();
        YearlyAverages yearly = computeYearlyAverages(data);
        double t1 = MPI_Wtime();
        log_event(rank, hostname, size, "compute_yearly", t0, t1);

        /* ---------- запись output.txt ---------- */
        t0 = MPI_Wtime();
        {
            std::ofstream out("output.txt");
            out << "Country,City,Year,AverageTemperature\n";

            for (const auto &cityPair : yearly) {
                const std::string &key = cityPair.first;
                const auto &yearMap = cityPair.second;

                for (const auto &yp : yearMap) {
                    size_t comma = key.find(',');
                    out << key.substr(0, comma) << ","
                        << key.substr(comma + 1) << ","
                        << yp.first << ","
                        << yp.second << "\n";
                }
            }
        }
        t1 = MPI_Wtime();
        log_event(rank, hostname, size, "write_output", t0, t1);

        /* ---------- ЭТАП 3: минимальные дельты ---------- */
        t0 = MPI_Wtime();
        auto deltas = computeMinDeltas(yearly);
        t1 = MPI_Wtime();
        log_event(rank, hostname, size, "compute_delta", t0, t1);

        /* ---------- запись min_delta.txt ---------- */
        t0 = MPI_Wtime();
        {
            std::ofstream out2("min_delta.txt");
            out2 << "Country,City,MinAbsYearlyDelta\n";

            int count = 0;
            for (const auto &md : deltas) {
                if (count >= 100) break;
                if (md.delta <= 0.0001) continue;

                size_t comma = md.key.find(',');
                out2 << md.key.substr(0, comma) << ","
                     << md.key.substr(comma + 1) << ","
                     << md.delta << "\n";
                count++;
            }
        }
        t1 = MPI_Wtime();
        log_event(rank, hostname, size, "write_min_delta", t0, t1);

        /* ---------- общее время rank 0 ---------- */
        log_event(rank, hostname, size,
                  "program_total",
                  t_program_start,
                  MPI_Wtime());
    }

    MPI_Finalize();
    return 0;
}