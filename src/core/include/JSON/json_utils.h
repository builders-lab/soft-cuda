#pragma once
#include <unordered_map>
#include <string>
#include "internal_header.h"
std::unordered_map<std::string, nlohmann::json>
readJsonToMap(const std::string& filePath);

std::unordered_map<std::string, json>
readDefaultToMap(std::string_view conf);
