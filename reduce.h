#pragma once
#include <vector>
#include "types.h"

std::vector<MinDelta>
reduceMinDeltasMPI(const std::vector<MinDelta> &local);
