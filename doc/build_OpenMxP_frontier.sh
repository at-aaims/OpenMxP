
source ../doc/load_modules_frontier.sh

SRC_DIR=../src

export LD_LIBRARY_PATH=/opt/rocm-5.1.0/llvm/lib:/opt/rocm-5.1.0/llvm/lib:${LD_LIBRARY_PATH}
export HIPCC_COMPILE_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND -std=c++14 -O3 -fopenmp --offload-arch=gfx90a"
rm -rf CMakeCache.txt CMakeFiles externals Makefile
cmake \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_BUILD_TYPE=Release \
    $SRC_DIR 2>&1 | tee CMAKE.OUTPUT
#   -DCMAKE_HIP_COMPILER_FORCED=True \

make VERBOSE=1 -j1 2>&1 | tee MAKE.OUTPUT
