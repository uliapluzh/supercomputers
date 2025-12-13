#include "reader.h"
#include <mpi.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

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

/* ---------- чтение диапазона строк ---------- */
static DataMap readCSVChunk(const std::string &filename,
                            std::size_t start,
                            std::size_t end)
{
    DataMap data;
    std::ifstream file(filename);
    if (!file) return data;

    std::string line;
    std::size_t line_no = 0;

    std::getline(file, line); // header
    line_no++;

    while (line_no < start && std::getline(file, line))
        line_no++;

    while (line_no < end && std::getline(file, line)) {
        line_no++;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string dt, tempStr, tempUncertStr, city, country, lat, lon;

        std::getline(ss, dt, ',');
        std::getline(ss, tempStr, ',');
        std::getline(ss, tempUncertStr, ',');
        std::getline(ss, city, ',');
        std::getline(ss, country, ',');
        std::getline(ss, lat, ',');
        std::getline(ss, lon, ',');

        if (dt.size() < 4 || city.empty() || country.empty())
            continue;

        tempStr.erase(
            std::remove_if(tempStr.begin(), tempStr.end(),
                           [](unsigned char c){ return std::isspace(c); }),
            tempStr.end()
        );

        if (tempStr.empty()) continue;

        double temp;
        try { temp = std::stod(tempStr); }
        catch (...) { continue; }

        int year;
        try { year = std::stoi(dt.substr(0,4)); }
        catch (...) { continue; }

        std::string key = country + "," + city;
        data[key].push_back({year, temp});
    }

    return data;
}

/* ---------- сериализация ---------- */
static std::string serialize(const DataMap &data)
{
    std::ostringstream out;
    for (auto &[key, vec] : data)
        for (auto &e : vec)
            out << key << ';' << e.year << ';' << e.temp << '\n';
    return out.str();
}

/* ---------- десериализация ---------- */
static DataMap deserialize(const std::string &buf)
{
    DataMap data;
    std::istringstream in(buf);
    std::string line;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string key, year, temp;

        std::getline(ss, key, ';');
        std::getline(ss, year, ';');
        std::getline(ss, temp, ';');

        data[key].push_back({std::stoi(year), std::stod(temp)});
    }
    return data;
}

/* ---------- MPI-чтение ---------- */
DataMap readCSV(const std::string &filename)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostlen;
    MPI_Get_processor_name(host, &hostlen);
    std::string hostname(host);

    double t_read_start = MPI_Wtime();

    std::size_t total_lines = 0;
    if (rank == 0) {
        std::ifstream f(filename);
        std::string l;
        while (std::getline(f, l)) total_lines++;
    }

    MPI_Bcast(&total_lines, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    std::size_t chunk = total_lines / size;
    std::size_t start = rank * chunk;
    std::size_t end   = (rank == size - 1) ? total_lines : start + chunk;

    double t0 = MPI_Wtime();
    DataMap local = readCSVChunk(filename, start, end);
    double t1 = MPI_Wtime();
    log_event(rank, hostname, size, "read_chunk", t0, t1);

    t0 = MPI_Wtime();
    std::string buf = serialize(local);
    t1 = MPI_Wtime();
    log_event(rank, hostname, size, "serialize", t0, t1);

    int len = buf.size();

    if (rank == 0) {
        DataMap result = local;

        t0 = MPI_Wtime();
        for (int src = 1; src < size; ++src) {
            MPI_Recv(&len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string recv(len, '\0');
            MPI_Recv(recv.data(), len, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            DataMap part = deserialize(recv);
            for (auto &[k, v] : part)
                result[k].insert(result[k].end(), v.begin(), v.end());
        }
        t1 = MPI_Wtime();
        log_event(rank, hostname, size, "comm", t0, t1);

        log_event(rank, hostname, size, "read_total",
                  t_read_start, MPI_Wtime());
        return result;
    }
    else {
        t0 = MPI_Wtime();
        MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(buf.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        log_event(rank, hostname, size, "comm", t0, t1);

        log_event(rank, hostname, size, "read_total",
                  t_read_start, MPI_Wtime());
        return {};
    }
}