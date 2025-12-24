#pragma once
#include "types.h"

PartialMap computeLocalPartials(const DataVec &data);
YearlyAverages computeFinalAverages(const PartialMap &local);
std::vector<MinDelta> computeMinDeltas(const YearlyAverages &yearly);
