cmake_minimum_required(VERSION 3.10)  # CMake version check
project(indiv)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 17)            # Enable c++14 standard

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES main.cpp)



# Add executable target with source files listed in SOURCE_FILES variable
add_executable(a.out ${SOURCE_FILES})

target_include_directories(a.out PUBLIC ./lemon-1.3.1/ ./lemon-1.3.1/build)
link_directories(a.out lemon-1.3.1/build/lemon)
target_link_libraries(a.out PUBLIC -lemon)

set (CMAKE_CXX_FLAGS "-O3 -Wall")
