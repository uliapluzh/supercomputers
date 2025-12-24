#include "reader.h"
#include "logging.h"

#include <mpi.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

DataVec readCSVChunk(const std::string &filename)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);
    std::string hostname(host);

    double t0 = MPI_Wtime();

    DataVec result;
    std::ifstream file(filename);
    if (!file) return result;

    std::string line;
    std::getline(file, line); // header

    std::size_t line_no = 0;

    while (std::getline(file, line)) {
        if (line_no % size != (std::size_t)rank) {
            ++line_no;
            continue;
        }
        ++line_no;

        std::stringstream ss(line);
        std::string dt, tempStr, uncertStr, city, country;

        std::getline(ss, dt, ',');
        std::getline(ss, tempStr, ',');
        std::getline(ss, uncertStr, ',');
        std::getline(ss, city, ',');
        std::getline(ss, country, ',');

        if (dt.size() < 4 || city.empty() || country.empty())
            continue;

        double uncert;
        try { uncert = std::stod(uncertStr); }
        catch (...) { continue; }

        if (uncert > 3.0) continue;

        double temp;
        try { temp = std::stod(tempStr); }
        catch (...) { continue; }

        int year = std::stoi(dt.substr(0,4));

        Record r;
        r.key  = country + "|" + city;
        r.year = year;
        r.temp = temp;

        result.push_back(r);
    }

    double t1 = MPI_Wtime();
    log_event(rank, hostname, size, "read+filter", t0, t1);

    return result;
}
