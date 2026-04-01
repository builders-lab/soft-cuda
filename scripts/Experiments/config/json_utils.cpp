#include "json_utils.h"
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

std::unordered_map<std::string, json>
readJsonToMap(const std::string& filePath) {

    std::unordered_map<std::string, json> result;

    // Open file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // Read JSON
    json j;
    file >> j;

    // Check if JSON is object
    if (!j.is_object()) {
        throw std::runtime_error("JSON root is not an object");
    }

    // Convert JSON to unordered_map
    for (auto& [key, value] : j.items()) {
        result[key] = value;
    }

    return result;
}
