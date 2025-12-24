#include "logging.h"
#include <fstream>

void log_event(int rank,
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