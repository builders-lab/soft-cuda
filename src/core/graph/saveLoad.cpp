#include "internal_header.h"

#include <fstream>
#include <cassert>
#include <string>

bool save_model(const std::string& filepath, const std::vector<tensor_t*>& weights) {
  std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;
    for (auto t : weights) {
        uint32_t num_elements = 1;
        uint8_t ndims = tensor_get_ndims(t);
        uint32_t* dims = tensor_get_dims(t);
        for (int i = 0; i < ndims; i++) {
            num_elements *= dims[i];
        }
        out.write(reinterpret_cast<const char*>(tensor_get_data(t)), num_elements * sizeof(float));
    }
    return true;
}

bool load_model(const std::string& filepath, const std::vector<tensor_t*>& weights) {
  std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;
    for (auto t : weights) {
        uint32_t num_elements = 1;
        uint8_t ndims = tensor_get_ndims(t);
        uint32_t* dims = tensor_get_dims(t);
        for (int i = 0; i < ndims; i++) {
            num_elements *= dims[i];
        }
        in.read(reinterpret_cast<char*>(tensor_get_data(t)), num_elements * sizeof(float));
    }
    return true;
}

