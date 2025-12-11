#include "reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

DataMap readCSV(const std::string &filename) {
    DataMap data;
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "ERROR: cannot open file " << filename << "\n";
        return data;
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);

        std::string dt, tempStr, tempUncertStr, city, country, lat, lon;

        std::getline(ss, dt, ',');
        std::getline(ss, tempStr, ',');
        std::getline(ss, tempUncertStr, ',');
        std::getline(ss, city, ',');
        std::getline(ss, country, ',');
        std::getline(ss, lat, ',');
        std::getline(ss, lon, ',');

        if (dt.size() < 4 || city.empty() || country.empty())
            continue;

        // --- ЧИСТИМ строку температуры ---
        tempStr.erase(
            std::remove_if(tempStr.begin(), tempStr.end(),
                           [](unsigned char c){ return std::isspace(c); }),
            tempStr.end()
        );

        // --- Пропускаем пустые температуры ---
        if (tempStr.empty())
            continue;

        // --- Парсим температуру ---
        double temp;
        try {
            temp = std::stod(tempStr);
        } catch (...) {
            continue; // если мусор
        }

        // --- Парсим год ---
        int year;
        try {
            year = std::stoi(dt.substr(0, 4));
        } catch (...) {
            continue;
        }

        // ключ = страна + город
        std::string key = country + "," + city;

        data[key].push_back({year, temp});
    }

    return data;
}