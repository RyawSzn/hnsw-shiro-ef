cmake_minimum_required(VERSION 3.10...3.26)

# Added 'C' to LANGUAGES because HDF5's CMake module usually looks for the C compiler too.
project(hnswlib LANGUAGES CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# 1. Use find_package to automatically locate apt-installed packages
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
# Require both C and CXX components for HDF5 so it finds libhdf5.so and libhdf5_cpp.so
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# 2. Setup hnswlib target
add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link Eigen and Boost to the hnswlib interface using their modern CMake targets
target_link_libraries(hnswlib INTERFACE Eigen3::Eigen Boost::boost)

# --- (Your compiler flags and examples section stays the same) ---
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()

if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET( CMAKE_CXX_FLAGS  "-O3 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
        if(COMPILER_SUPPORT_NATIVE_FLAG)
            SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
            message("set -march=native flag")
        else()
            check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
            if(COMPILER_SUPPORT_M1_FLAG)
                SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
                message("set -mcpu=apple-m1 flag")
            endif()
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET( CMAKE_CXX_FLAGS  "-O3 -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=5" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()
endif()
# ------------------------------------------------------------------

# 3. Setup your custom run executable
add_executable(run experiments_driver/run.cpp)

# ef_sweep: sweeps ef_max 50..4000 and logs latency + recall per dataset
add_executable(ef_sweep research/ef_sweep.cpp)

# Add HDF5 include directories (found automatically by find_package)
target_include_directories(run PRIVATE ${HDF5_INCLUDE_DIRS})

# 4. Link everything together correctly
target_link_libraries(run PRIVATE
    hnswlib                 # Link your hnswlib library (you forgot this in your script!)
    Eigen3::Eigen           # Link Eigen
    Boost::boost            # Link Boost
    ${HDF5_CXX_LIBRARIES}   # Link libhdf5_cpp.so
    ${HDF5_LIBRARIES}       # Link libhdf5.so
)

# ef_sweep shares the same deps as run
target_include_directories(ef_sweep PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(ef_sweep PRIVATE
    hnswlib
    Eigen3::Eigen
    Boost::boost
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

# gamma_sweep: sweeps the gamma (decay constant) across all 3 datasets and logs
# structured [GAMMA_SWEEP] lines that visualize_gamma_sweep.py can plot.
add_executable(gamma_sweep research/gamma_sweep.cpp)
target_include_directories(gamma_sweep PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gamma_sweep PRIVATE
    hnswlib
    Eigen3::Eigen
    Boost::boost
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

# gamma_correlation: outputs RV vs Recall CSV for different gamma values
add_executable(gamma_correlation research/gamma_correlation.cpp)
target_include_directories(gamma_correlation PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gamma_correlation PRIVATE
    hnswlib
    Eigen3::Eigen
    Boost::boost
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

# gamma_buckets: exports the EfAdapter lookup table mapping (Score -> EF)
add_executable(gamma_buckets research/gamma_buckets.cpp)
target_include_directories(gamma_buckets PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gamma_buckets PRIVATE
    hnswlib
    Eigen3::Eigen
    Boost::boost
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)
