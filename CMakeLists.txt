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

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hbfine/CMakeLists.txt")
  message(FATAL_ERROR "HBFine submodule is missing; initialize 3rdparty/hbfine before building")
endif()
add_subdirectory(3rdparty/hbfine)

add_library(gtsim_core STATIC
  src/scheduler.cpp
  src/subpartition.cpp
  src/sm.cpp
  src/gpu.cpp
  src/memory/hbfine.cpp
  src/memory/base.cpp
  src/memory/address.cpp
  src/memory/naive.cpp
  src/memory/system.cpp
  src/memory/sram.cpp
  src/simulator.cpp
  src/runtime/kernel_execution_session.cpp)
target_include_directories(gtsim_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glaze/include)

target_link_libraries(gtsim_core PUBLIC HBFine::hbfine)

include(cmake/HBFBackends.cmake)

option(GTSIM_ENABLE_RAMULATOR2 "Include the runtime-selectable Ramulator2 HBM3 backend" ON)
if(GTSIM_ENABLE_RAMULATOR2)
  include(cmake/Ramulator2.cmake)
endif()

add_executable(gtsim_hbf_layout tools/hbf_layout.cpp)
target_link_libraries(gtsim_hbf_layout PRIVATE gtsim_core)

add_executable(gtsim_gemm workloads/gemm/gemm_h100.cpp)
target_link_libraries(gtsim_gemm PRIVATE gtsim_core)

add_executable(gtsim_fa3 workloads/fa3/fa3_h100.cpp)
target_link_libraries(gtsim_fa3 PRIVATE gtsim_core)

add_executable(gtsim_fa2 workloads/fa2/fa2_a100.cpp)
target_link_libraries(gtsim_fa2 PRIVATE gtsim_core)

option(GTSIM_BUILD_IR_FRAMEWORK "Build the layered IR framework and its tests" ON)
if(GTSIM_BUILD_IR_FRAMEWORK)
  add_library(gtsim_execution STATIC
    src/execution_ir/kernel_graph_builder.cpp
    src/runtime/stream_protocol.cpp
    src/runtime/streaming_program_runner.cpp)
  target_link_libraries(gtsim_execution PUBLIC gtsim_core)
  add_executable(gtsim_program workloads/serve/program_runner.cpp)
  target_link_libraries(gtsim_program PRIVATE gtsim_execution)
endif()


include(CTest)
if(BUILD_TESTING)
  add_executable(hbf_prefetch_check tests/hbf_prefetch.cpp)
  target_link_libraries(hbf_prefetch_check PRIVATE gtsim_core)
  add_executable(flashaccel_partition_oracle tests/flashaccel_partition_oracle.cpp)
  add_test(NAME flashaccel_partition_oracle COMMAND flashaccel_partition_oracle)
  add_executable(hbf_grouped_geometry_check tests/hbf_grouped_geometry.cpp)
  target_link_libraries(hbf_grouped_geometry_check PRIVATE gtsim_core)
  if(GTSIM_BUILD_IR_FRAMEWORK)
    add_executable(reference_graph_dump tests/reference_parity/reference_graph_dump.cpp)
    target_link_libraries(reference_graph_dump PRIVATE gtsim_core)
  endif()
  set(H100_ARCH_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/h100-sxm.yaml)
  set(MEMORY_FULL_HBM_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_full_hbm-0F5M.yaml)
  set(MEMORY_FULL_HBF_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_full_hbf-5F0M.yaml)
  set(MEMORY_CO_LOCATED_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_co_located-3F2M.yaml)
  set(MEMORY_CASCADED_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_cascaded-3F2M.yaml)
  add_executable(fa2_tests tests/fa2_tests.cpp)
  target_link_libraries(fa2_tests PRIVATE gtsim_core)
  add_test(NAME fa2_decode_semantics COMMAND fa2_tests
           ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/a100-pcie40-simple.yaml)
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
  add_executable(kernel_swopt_tests tests/kernel_swopt_tests.cpp)
  target_link_libraries(kernel_swopt_tests PRIVATE gtsim_core)
  add_test(NAME kernel_swopt_semantics COMMAND kernel_swopt_tests)
  add_executable(completion_queue_tests tests/completion_queue_tests.cpp)
  target_link_libraries(completion_queue_tests PRIVATE gtsim_core)
  add_test(NAME completion_queue_semantics COMMAND completion_queue_tests)
  add_executable(hbm_address_tests tests/hbm_address_tests.cpp)
  target_link_libraries(hbm_address_tests PRIVATE gtsim_core)
  add_test(NAME hbm_address_and_clock COMMAND hbm_address_tests)
  add_executable(hbf_serving_tests tests/hbf_serving_tests.cpp)
  target_link_libraries(hbf_serving_tests PRIVATE gtsim_core)
  add_test(NAME hbf_serving_matched_parameters COMMAND hbf_serving_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/a100-pcie40-3m2f-grade1-simple-matched.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/a100-pcie40-3m2f-grade1-hbfine.yaml)
  add_executable(hbf_backend_tests tests/hbf_backend_tests.cpp)
  target_link_libraries(hbf_backend_tests PRIVATE gtsim_core)
  add_test(NAME hbf_bank_plane_contract COMMAND hbf_backend_tests)
  add_executable(workload_placement_tests tests/workload_placement_tests.cpp
    workloads/gemm/gemm_h100.cpp workloads/fa3/fa3_h100.cpp)
  target_compile_definitions(workload_placement_tests PRIVATE GTSIM_WORKLOAD_LIBRARY)
  target_link_libraries(workload_placement_tests PRIVATE gtsim_core)
  add_test(NAME mixed_workload_placement COMMAND workload_placement_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/memconf/3F2M_bank_plane.yaml)
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
    if(GTSIM_BUILD_IR_FRAMEWORK)
      add_test(NAME gemm_buffer_sync
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_parity/audit_gemm_sync.py
                $<TARGET_FILE:reference_graph_dump> --require-safe
                --output ${CMAKE_CURRENT_BINARY_DIR}/gemm-sync-audit.json)
    endif()
    add_test(NAME mixed_hbfine_smoke
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/hbf/check_smoke.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(mixed_hbfine_smoke PROPERTIES TIMEOUT 180)
    add_test(NAME workload_profile_boundary
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_workload_profile.py)
    add_test(NAME v0_1_0_regression
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_reference.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR} --verify)
    if(GTSIM_BUILD_IR_FRAMEWORK)
      add_test(NAME streaming_lifecycle
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/streaming_lifecycle.py
                --build-dir ${CMAKE_CURRENT_BINARY_DIR})
      set_tests_properties(streaming_lifecycle PROPERTIES TIMEOUT 180)
    endif()
  endif()
endif()

install(TARGETS gtsim_core gtsim_gemm gtsim_fa3 gtsim_fa2
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/gtsim)
install(DIRECTORY 3rdparty/glaze/include/glaze DESTINATION include)

if(GTSIM_BUILD_IR_FRAMEWORK)
  install(TARGETS gtsim_program RUNTIME DESTINATION bin)
endif()
if(BUILD_TESTING)
  add_executable(hbf_online_contract_tests tests/hbf_online_contract_tests.cpp)
  target_link_libraries(hbf_online_contract_tests PRIVATE gtsim_core flashlab_core HBFEventLab::core)
  add_test(NAME hbf_online_contract COMMAND hbf_online_contract_tests)
endif()
