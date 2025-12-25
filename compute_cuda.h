#pragma once
#include <vector>

extern "C" {

// ЧИСТЫЕ C-типы на границе
std::vector<double>
computeMinDeltasCUDA(
    const std::vector<double> &values,
    const std::vector<int> &offsets,
    const std::vector<int> &lengths
);

}
