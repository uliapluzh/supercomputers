#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <map>

struct Entry {
    int year;
    double temp;
};

using DataMap = std::map<std::string, std::vector<Entry>>;

#endif
