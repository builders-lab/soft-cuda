#include <iostream>
#include "json_utils.h"

int main() {
    try {
        auto data = readJsonToMap("data.json");

        for (const auto& [key, value] : data) {
            std::cout << key << " : " << value << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}
