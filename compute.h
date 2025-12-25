#pragma once
#include "types.h"

bool canUseCUDA();
PartialMap computeLocalPartials(const DataVec &data);
YearlyAverages computeFinalAverages(const PartialMap &local);
std::vector<MinDelta> computeMinDeltas(const YearlyAverages &yearly);
struct FlatSeries {
    std::string key;
    std::vector<double> values; // средние по годам, отсортированы
};

std::vector<MinDelta>
computeMinDeltasCPU(const std::vector<FlatSeries> &series);

std::vector<FlatSeries>
flattenYearlyAverages(const YearlyAverages &yearly);
