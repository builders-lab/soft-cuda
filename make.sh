#!/bin/bash

current_dir=$(pwd)

usage() {
  echo -e "Usage: $(basename "$0") [flags]\n"
  echo "  -b   Build (debug)"
  echo "  -z   Clean build (remove build/ first)"
  echo "  -x   Clean build (interactive, prompts before deleting)"
  echo "  -r   Run (stderr suppressed)"
  echo "  -v   Run (verbose, with timing)"
  echo "  -t   Run tests (ctest)"
  echo "  -m   Run benchmarks"
  echo "  -h   Show this help"
}

build() {
  time cmake -B build -DCMAKE_BUILD_TYPE=Debug
  time cmake --build build -j$(nproc)
  builtin  echo "===== BUILD COMPLETE ====="
}

while getopts "brhvtzxm" opt; do
  case "$opt" in
    h) usage ;;
    b) build ;;
    z|x)
      if [[ -d "build" && $opt == "x" ]]; then
        rm -rfi "${current_dir}/build/"
      else
        rm -rf "${current_dir}/build/"
      fi
      build
      ;;
    v) echo "===== RUNNING ====="
       time "${current_dir}/build/soft" ;;
    r) time "${current_dir}/build/soft" 2>/dev/null ;;
    t) cd build && ctest ;;
    m) echo "===== BENCHMARKS ====="
       "${current_dir}/benchmarks/run_all.sh" 2>/dev/null ;;
   \?) usage; exit 1 ;;
  esac
done

if [[ $OPTIND -eq 1 ]]; then
  usage
fi
