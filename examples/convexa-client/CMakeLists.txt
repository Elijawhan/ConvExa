cmake_minimum_required(VERSION 3.15)
project(ClientProject)

find_package(convexa REQUIRED)

add_executable(client_executable main.cpp)
target_link_libraries(client_executable PRIVATE convexa::convexa)
