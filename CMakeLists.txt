cmake_minimum_required(VERSION 3.14)
project(trustworthiness_dim_reduction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen: use installed if available, otherwise fetch
find_package(Eigen3 QUIET NO_MODULE)

include(FetchContent)
if (NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

add_executable(trustworthiness_dim_reduction
    src/main.cpp
    src/utils.cpp
    src/preservetopo.cpp
    src/pca.cpp
    src/kmeans.cpp
    data/datagen.cpp
)

target_include_directories(trustworthiness_dim_reduction PRIVATE src data)
target_link_libraries(trustworthiness_dim_reduction PRIVATE Eigen3::Eigen)
target_compile_definitions(trustworthiness_dim_reduction PRIVATE _USE_MATH_DEFINES _ENFORCE_MATCHING_ALLOCATORS=0)
