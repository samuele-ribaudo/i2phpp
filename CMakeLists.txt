cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

project(i2phpp)

# find MPI
find_package(MPI COMPONENTS C CXX REQUIRED)

include(FetchContent)

# json library
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# plot library
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master)
FetchContent_MakeAvailable(matplotplusplus)

# Set DEBUG macro if project is build in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

set(lib_name "i2phpp_lib")
file(GLOB_RECURSE sources "${CMAKE_SOURCE_DIR}/source/**/*.cpp")

add_library(${lib_name} ${sources})

# include all required directories
target_include_directories(${lib_name}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(applications)

target_link_libraries(${lib_name} PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(${lib_name} PUBLIC matplot)
target_link_libraries(${lib_name} PUBLIC MPI::MPI_CXX)

# tests
option(WITH_TESTS "Configure tests" ON)
if (WITH_TESTS)
    message(STATUS "Tests: enabled")
    # Deactivate internal benchmark tests
    set(BENCHMARK_ENABLE_TESTING OFF)
    # Ensure that google benchmark debug assertions are disabled
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.16.0)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
else ()
    message(STATUS "Tests: disabled")
endif ()
