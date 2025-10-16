cmake_minimum_required(VERSION 3.8)
project(armadillo_practice)

find_path(
    mlpack_dir 
    NAMES "mlpack.hpp"
    PATHS ../.. C:/Users/Shayan/vcpkg/packages/mlpack_x64-windows/include
)
find_path(
    armadillo_dir
    NAMES "armadillo"
    PATHS ../.. C:/Users/Shayan/vcpkg/packages/Armadillo/include
)
find_path(
    blas_dir 
    NAMES "openblas.lib"
    PATHS ../.. C:/Users/Shayan/vcpkg/packages/openblas_x64-windows/lib
    # PATHS ../.. C:/Users/Shayan/vcpkg/packages/openblas_x64-windows/share/openblas
    # PATHS ../.. C:/Users/Shayan/vcpkg/installed/x64-windows/lib
)
find_path(
    lapack_dir 
    NAMES "lapack.lib"
    PATHS ../.. C:/Users/Shayan/vcpkg/packages/lapack-reference_x64-windows/lib
    # PATHS ../.. C:/Users/Shayan/vcpkg/installed/x64-windows/lib
)
find_path(
    ensmallen_dir 
    NAMES "ensmallen.hpp"
    PATHS ../.. C:/Users/Shayan/vcpkg/packages/ensmallen_x64-windows/include
)
find_package(OpenBLAS 0.3.29 EXACT CONFIG REQUIRED)
find_package(lapack CONFIG REQUIRED
    PATHS ../.. C:/Users/Shayan/vcpkg/buildtrees/lapack-reference/x64-windows-dbg
)

# add the executable
add_executable(practice test.cpp)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(practice PUBLIC
                            # "${PROJECT_BINARY_DIR}"
                            "${mlpack_dir}"
                            "${armadillo_dir}"
                            # "${cereal_dir}"
                            "${ensmallen_dir}"
                            "${blas_dir}"
                            "${lapack_dir}"
                            OpenBLAS::OpenBLAS
                            LAPACK::LAPACK
                            )