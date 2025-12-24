#pragma once

#include <string>
#include <vector>
#include <map>

struct Record {
    std::string key;   // "Country|City"
    int year;
    double temp;
};

struct Stat {
    double sum = 0.0;
    int count = 0;
};

using DataVec = std::vector<Record>;

using PartialMap =
    std::map<std::string,
        std::map<int, Stat>>;

using YearlyAverages =
    std::map<std::string,
        std::map<int, double>>;

struct MinDelta {
    std::string key;
    double delta;
};
