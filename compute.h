#pragma once

#include <vector>
#include "types.h"

// локальная статистика (CPU или GPU — внутри)
std::vector<MinDelta>
computeLocalStats(const DataVec &data);

// CPU-части (используются как fallback и в тестах)
PartialMap computeLocalPartials(const DataVec &data);
YearlyAverages computeFinalAverages(const PartialMap &p);
std::vector<MinDelta> computeMinDeltasCPU(const YearlyAverages &yearly);

// проверка, можно ли использовать CUDA на этом rank
bool canUseCUDA();
