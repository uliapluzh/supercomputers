#include "compute.h"
#include "types.h"
#include <algorithm>
#include <limits>

YearlyAverages computeYearlyAverages(const DataMap &data) {
    YearlyAverages result;

    for (const auto &pair : data) {
        const std::string &key = pair.first;            // Country,City
        const std::vector<Entry> &entries = pair.second;

        std::map<int, std::vector<double>> byYear;

        // Группируем температуры по годам
        for (const Entry &e : entries) {
            byYear[e.year].push_back(e.temp);
        }

        // Считаем среднее по каждому году
        for (const auto &y : byYear) {
            int year = y.first;
            const std::vector<double> &temps = y.second;

            if (temps.empty()) continue;

            double sum = 0.0;
            for (double t : temps) sum += t;
            double avg = sum / temps.size();

            result[key][year] = avg;
        }
    }

    return result;
}


std::vector<MinDelta> computeMinDeltas(const YearlyAverages &yearly) {
    std::vector<MinDelta> result;

    for (const auto &pair : yearly) {
        const std::string &key = pair.first;
        const auto &ymap = pair.second;

        if (ymap.size() < 2) continue;

        double minDelta = std::numeric_limits<double>::max();

        // предыдущий год
        auto it = ymap.begin();
        auto prev = it++;
        for (; it != ymap.end(); ++it, ++prev) {
            double d = std::abs(it->second - prev->second);
            if (d < 1e-7) d = 0.0;
            if (d < minDelta) minDelta = d;
        }

        result.push_back({key, minDelta});
    }

    // сортируем по возрастанию дельты
    std::sort(result.begin(), result.end(),
              [](const MinDelta &a, const MinDelta &b) {
                  return a.delta < b.delta;
              });

    return result;
}