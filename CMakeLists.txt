# Version CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(fcn_model_image_segmentation)

# Option
set(CMAKE_CXX_STANDARD 17)

# Compiler-specific options
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_compile_options(
        -Werror
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Source files
file(GLOB_RECURSE SOURCES
    source/*.cpp
)

find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Boost components
set(BOOST_COMPONENTS
    program_options
    system
    filesystem
    date_time
    chrono
    serialization
)
find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)

# Compiling target
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    stdc++fs
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
