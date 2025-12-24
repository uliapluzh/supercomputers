#include "compute.h"
#include "logging.h"

#include <mpi.h>
#include <algorithm>
#include <limits>

PartialMap computeLocalPartials(const DataVec &data)
{
    PartialMap p;
    for (const auto &r : data) {
        auto &s = p[r.key][r.year];
        s.sum   += r.temp;
        s.count += 1;
    }
    return p;
}

YearlyAverages computeFinalAverages(const PartialMap &p)
{
    YearlyAverages y;
    for (const auto &[key, years] : p)
        for (const auto &[year, stat] : years)
            y[key][year] = stat.sum / stat.count;
    return y;
}

std::vector<MinDelta> computeMinDeltas(const YearlyAverages &yearly)
{
    std::vector<MinDelta> res;

    for (const auto &[key, ym] : yearly) {
        if (ym.size() < 2) continue;

        double best = std::numeric_limits<double>::max();
        auto it = ym.begin();
        auto prev = it++;

        for (; it != ym.end(); ++it, ++prev)
            best = std::min(best, std::abs(it->second - prev->second));

        res.push_back({key, best});
    }

    std::sort(res.begin(), res.end(),
              [](auto &a, auto &b){ return a.delta < b.delta; });

    return res;
}
