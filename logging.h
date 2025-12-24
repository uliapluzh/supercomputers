#pragma once
#include <string>

void log_event(int rank,
               const std::string &host,
               int size,
               const std::string &op,
               double t0,
               double t1);
