cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed convexa library
find_package(convexa REQUIRED)

# Add your executable
add_executable(my_project main.cpp)

# Link convexa to your executable
target_link_libraries(my_project PRIVATE convexa::convexa)
