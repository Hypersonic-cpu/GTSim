include(ExternalProject)
include(ProcessorCount)
ProcessorCount(gtsim_available_cores)
if(NOT gtsim_available_cores)
  set(gtsim_available_cores 1)
endif()
set(GTSIM_NATIVE_BUILD_JOBS "${gtsim_available_cores}" CACHE STRING
    "Parallel jobs for the isolated native memory library build")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(eventlab_patched_source "${CMAKE_CURRENT_BINARY_DIR}/eventlab-source")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-sram.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-megaplane.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-channel-admission.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-stream-retirement.patch")
execute_process(COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/memory/prepare_native.py"
  --source "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hbf-eventlab"
  --output "${eventlab_patched_source}"
  --patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-sram.patch"
  --patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-megaplane.patch"
  --patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-channel-admission.patch"
  --patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/eventlab-stream-retirement.patch"
  COMMAND_ERROR_IS_FATAL ANY)
add_subdirectory("${eventlab_patched_source}" "${CMAKE_CURRENT_BINARY_DIR}/eventlab-integrated" EXCLUDE_FROM_ALL)
set(flashlab_patched_source "${CMAKE_CURRENT_BINARY_DIR}/flashlab-source")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/flashlab-sram.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/flashlab-megaplane.patch"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/memory/prepare_native.py")
execute_process(COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/memory/prepare_native.py"
  --source "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/flashlab-sim"
  --output "${flashlab_patched_source}"
  --patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/flashlab-sram.patch"
  --patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/flashlab-megaplane.patch"
  COMMAND_ERROR_IS_FATAL ANY)

ExternalProject_Add(flashlab_native
  SOURCE_DIR "${flashlab_patched_source}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/flashlab-integrated"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target hbfsim_core --parallel ${GTSIM_NATIVE_BUILD_JOBS}
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/flashlab-integrated/libhbfsim_core.a")
add_library(flashlab_core STATIC IMPORTED)
set_target_properties(flashlab_core PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/flashlab-integrated/libhbfsim_core.a"
  INTERFACE_INCLUDE_DIRECTORIES "${flashlab_patched_source}/src")
add_dependencies(flashlab_core flashlab_native)
target_sources(gtsim_core PRIVATE src/memory/eventlab.cpp src/memory/flashlab.cpp)
target_link_libraries(gtsim_core PRIVATE HBFEventLab::core flashlab_core)
