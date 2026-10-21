cmake_minimum_required(VERSION 3.20)
project(GTSim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(gtsim_core STATIC
  src/scheduler.cpp
  src/subpartition.cpp
  src/sm.cpp
  src/gpu.cpp
  src/memory_backend.cpp
  src/simulator.cpp)
target_include_directories(gtsim_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glaze/include)

option(GTSIM_ENABLE_RAMULATOR2 "Build the optional Ramulator2 HBM3 backend" OFF)
if(GTSIM_ENABLE_RAMULATOR2)
  include(cmake/Ramulator2.cmake)
endif()

add_executable(gtsim_gemm workloads/gemm/gemm_h100.cpp)
target_link_libraries(gtsim_gemm PRIVATE gtsim_core)

add_executable(gtsim_fa3 workloads/fa3/fa3_h100.cpp)
target_link_libraries(gtsim_fa3 PRIVATE gtsim_core)

include(CTest)
if(BUILD_TESTING)
  set(H100_ARCH_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/h100-sxm.yaml)
  set(MEMORY_FULL_HBM_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_full_hbm-0F5M.yaml)
  set(MEMORY_FULL_HBF_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_full_hbf-5F0M.yaml)
  set(MEMORY_CO_LOCATED_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_co_located-3F2M.yaml)
  set(MEMORY_CASCADED_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_cascaded-3F2M.yaml)
  add_test(NAME gemm_smoke COMMAND gtsim_gemm ${H100_ARCH_CONFIG}
           --memory-config ${MEMORY_FULL_HBM_CONFIG} 512 512 512)
  add_test(NAME fa3_smoke COMMAND gtsim_fa3 ${H100_ARCH_CONFIG}
           --memory-config ${MEMORY_FULL_HBM_CONFIG} 1 1 1 128 128)
  add_executable(arch_conf_tests tests/arch_conf_tests.cpp)
  target_link_libraries(arch_conf_tests PRIVATE gtsim_core)
  add_test(NAME arch_conf_parser COMMAND arch_conf_tests ${H100_ARCH_CONFIG}
           ${MEMORY_FULL_HBM_CONFIG} ${MEMORY_FULL_HBF_CONFIG}
           ${MEMORY_CO_LOCATED_CONFIG} ${MEMORY_CASCADED_CONFIG})
  add_executable(memory_address_tests tests/memory_address_tests.cpp)
  target_link_libraries(memory_address_tests PRIVATE gtsim_core)
  add_test(NAME memory_address_identity COMMAND memory_address_tests)
  add_executable(memory_topology_tests tests/memory_topology_tests.cpp)
  target_link_libraries(memory_topology_tests PRIVATE gtsim_core)
  add_test(NAME memory_topology_routes COMMAND memory_topology_tests)
  add_executable(completion_queue_tests tests/completion_queue_tests.cpp)
  target_link_libraries(completion_queue_tests PRIVATE gtsim_core)
  add_test(NAME completion_queue_semantics COMMAND completion_queue_tests)
  add_executable(hbm_address_tests tests/hbm_address_tests.cpp)
  target_link_libraries(hbm_address_tests PRIVATE gtsim_core)
  add_test(NAME hbm_address_and_clock COMMAND hbm_address_tests)
  if(GTSIM_ENABLE_RAMULATOR2)
    set(RAMULATOR_H100_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/ramulator/h100-sxm-ramulator.yaml)
    add_executable(ramulator_backend_tests tests/ramulator_backend_tests.cpp)
    target_link_libraries(ramulator_backend_tests PRIVATE gtsim_core)
    add_test(NAME ramulator_backend_contract COMMAND ramulator_backend_tests ${RAMULATOR_H100_CONFIG})
    add_test(NAME ramulator_gemm_smoke COMMAND gtsim_gemm ${H100_ARCH_CONFIG}
             --memory-config ${RAMULATOR_H100_CONFIG} 128 256 64)
    add_test(NAME ramulator_fa3_smoke COMMAND gtsim_fa3 ${H100_ARCH_CONFIG}
             --memory-config ${RAMULATOR_H100_CONFIG} 1 1 1 128 128)
    set_tests_properties(ramulator_backend_contract ramulator_gemm_smoke ramulator_fa3_smoke
                         PROPERTIES TIMEOUT 180)
  endif()
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME workload_profile_boundary
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_workload_profile.py)
    add_test(NAME v0_1_0_regression
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_reference.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR} --verify)
  endif()
endif()

install(TARGETS gtsim_core gtsim_gemm gtsim_fa3
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/gtsim)
install(DIRECTORY 3rdparty/glaze/include/glaze DESTINATION include)
