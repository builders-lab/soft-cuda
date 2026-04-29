#include "internal_header.h"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : nullptr;
    printf("||==============================================||\n");
    printf("||  soft-cuda AOT Hardware Profiler             ||\n");
    printf("||==============================================||\n\n");
    return soft_profile_and_write(path);
}
