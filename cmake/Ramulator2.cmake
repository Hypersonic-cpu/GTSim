# Build the registered submodule without rewriting its checkout. The current
# ParamReader header omits a required dependent-template disambiguator; create
# a one-line compatibility overlay until that spelling is fixed upstream.
set(ramulator_source "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ramulator2")
set(ramulator_overlay "${CMAKE_CURRENT_BINARY_DIR}/ramulator-compat")
file(READ "${ramulator_source}/src/ramulator/base/param.h" ramulator_param)
string(REPLACE "config[name].as<T>()" "config[name].template as<T>()"
       ramulator_param "${ramulator_param}")
file(MAKE_DIRECTORY "${ramulator_overlay}/ramulator/base")
file(CONFIGURE OUTPUT "${ramulator_overlay}/ramulator/base/param.h"
     CONTENT "${ramulator_param}" @ONLY)
include_directories(BEFORE "${ramulator_overlay}")

# fmt 10's consteval format checking fails with AppleClang 21/libc++.
# Its supported FMT_CONSTEVAL override keeps runtime formatting unchanged.
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21)
  add_compile_definitions(FMT_CONSTEVAL=)
endif()

set(RAMULATOR_PYTHON_BINDINGS OFF CACHE BOOL "Build Ramulator Python bindings" FORCE)
add_subdirectory(3rdparty/ramulator2 EXCLUDE_FROM_ALL)
set_target_properties(ramulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
target_sources(gtsim_core PRIVATE src/ramulator_backend.cpp)
target_include_directories(gtsim_core SYSTEM PRIVATE "${ramulator_source}/src")
target_compile_definitions(gtsim_core PRIVATE GTSIM_ENABLE_RAMULATOR2)
target_link_libraries(gtsim_core PRIVATE ramulator)
