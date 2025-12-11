#ifndef COMPUTE_H
#define COMPUTE_H

#include "types.h"
#include <map>
#include <string>
#include <vector>
#include <cmath>

using YearlyAverages = std::map<std::string, std::map<int, double>>;

YearlyAverages computeYearlyAverages(const DataMap &data);

struct MinDelta {
    std::string key;   // Country,City
    double delta;
};

std::vector<MinDelta> computeMinDeltas(const YearlyAverages &yearly);

#endif
