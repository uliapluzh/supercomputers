#include <iostream>
#include <fstream>
#include "reader.h"
#include "compute.h"

int main() {
    std::string filename = "GlobalLandTemperaturesByCity.csv";

    DataMap data = readCSV(filename);
    YearlyAverages yearly = computeYearlyAverages(data);

    std::ofstream out("output.txt");
    if (!out) {
        std::cerr << "ERROR: cannot write to output.txt\n";
        return 1;
    }

    out << "Country,City,Year,AverageTemperature\n";

    for (const auto &cityPair : yearly) {
        // cityPair.first = "Country,City"
        const std::string &key = cityPair.first;
        const auto &yearMap = cityPair.second;

        for (const auto &yp : yearMap) {
            int year = yp.first;
            double avg = yp.second;

            // Разбиваем Country,City
            size_t comma = key.find(',');
            std::string country = key.substr(0, comma);
            std::string city = key.substr(comma + 1);

            out << country << "," << city << ","
                << year << "," << avg << "\n";
        }
    }

    std::cout << "Результат записан в output.txt\n";

    // ====== ЭТАП 2: MinAbsYearlyDelta ======
    auto deltas = computeMinDeltas(yearly);

    std::ofstream out2("min_delta.txt");
    if (!out2) {
        std::cerr << "ERROR: cannot write to min_delta.txt\n";
        return 1;
    }

    out2 << "Country,City,MinAbsYearlyDelta\n";

    int count = 0;
    for (const auto &md : deltas) {
        if (count >= 100) break;
        if (md.delta <= 0.0001) continue;

        size_t comma = md.key.find(',');
        std::string country = md.key.substr(0, comma);
        std::string city = md.key.substr(comma + 1);

        out2 << country << "," << city << "," << md.delta << "\n";
        count++;
    }

    std::cout << "Результат записан в min_delta.txt (ТОП-50)\n";

    return 0;
}