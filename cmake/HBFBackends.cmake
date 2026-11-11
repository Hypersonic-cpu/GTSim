add_subdirectory(3rdparty/hbf-eventlab EXCLUDE_FROM_ALL)
include(ExternalProject)
ExternalProject_Add(flashlab_native
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/flashlab-sim"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/flashlab-native"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target hbfsim_core --parallel 4
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/flashlab-native/libhbfsim_core.a")
add_library(flashlab_core STATIC IMPORTED)
set_target_properties(flashlab_core PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/flashlab-native/libhbfsim_core.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/flashlab-sim/src")
add_dependencies(flashlab_core flashlab_native)
target_sources(gtsim_core PRIVATE src/memory/eventlab.cpp src/memory/flashlab.cpp)
target_link_libraries(gtsim_core PRIVATE HBFEventLab::core flashlab_core)
