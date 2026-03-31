#pragma once
#include <unordered_map>
#include <string>
#include "json.hpp"

std::unordered_map<std::string, nlohmann::json>
readJsonToMap(const std::string& filePath);