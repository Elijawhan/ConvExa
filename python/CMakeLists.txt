cmake_minimum_required(VERSION 3.15)
project(convexa)

find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(convexa REQUIRED)



add_library(convexa_core MODULE bindings.cpp)
target_link_libraries(convexa_core PRIVATE pybind11::module Python3::Python
  convexa::convexa)


set_target_properties(convexa_core PROPERTIES PREFIX "")
set_target_properties(convexa_core PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

# On macOS, add @rpath to convexa_core to find libconvexa
if(APPLE)
    set_target_properties(convexa_core PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
endif()
# Install the convexa module to the correct location
# Determine the Python site-packages directory
set(PYTHON_SITE_PACKAGES "${Python3_SITELIB}")

install(TARGETS convexa_core
  DESTINATION ${PYTHON_SITE_PACKAGES}/convexa) # Important install line.

get_target_property(CONVEXA_LIBRARY convexa::convexa LOCATION)
install(FILES ${CONVEXA_LIBRARY}
    DESTINATION ${PYTHON_SITE_PACKAGES}/convexa
)
