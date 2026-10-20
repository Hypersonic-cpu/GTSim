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
  src/simulator.cpp)
target_include_directories(gtsim_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glaze/include)

add_executable(gtsim_gemm workloads/gemm/gemm_h100.cpp)
target_link_libraries(gtsim_gemm PRIVATE gtsim_core)

add_library(gtsim_tile_ir STATIC src/tile_ir_parser.cpp src/tile_dag_builder.cpp)
target_link_libraries(gtsim_tile_ir PUBLIC gtsim_core)
add_executable(gtsim_gemm_ir workloads/gemm/gemm_ir.cpp)
target_link_libraries(gtsim_gemm_ir PRIVATE gtsim_tile_ir)

add_executable(gtsim_fa3 workloads/fa3/fa3_h100.cpp)
target_link_libraries(gtsim_fa3 PRIVATE gtsim_core)

include(CTest)
set(GTSIM_TVM_PYTHON "" CACHE FILEPATH "Python interpreter with the separately built pinned TVM (optional integration tests)")
if(BUILD_TESTING)
  set(H100_ARCH_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/h100-sxm.yaml)
  set(MEMORY_FULL_HBM_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_full_hbm-0F5M.yaml)
  set(MEMORY_FULL_HBF_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_full_hbf-5F0M.yaml)
  set(MEMORY_CO_LOCATED_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_co_located-3F2M.yaml)
  set(MEMORY_CASCADED_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/configs/v0.1.0/memory_cascaded-3F2M.yaml)
  add_executable(tile_ir_tests tests/tile_ir_tests.cpp)
  target_link_libraries(tile_ir_tests PRIVATE gtsim_tile_ir)
  add_executable(tile_ir_parity_tests tests/tile_ir_parity_tests.cpp)
  target_link_libraries(tile_ir_parity_tests PRIVATE gtsim_tile_ir)
  add_test(NAME tile_ir_gemm_parity COMMAND tile_ir_parity_tests
           ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/h100_gemm.gtsir 128 256 128)
  add_test(NAME tile_ir_boundary COMMAND tile_ir_tests
           ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/copy.gtsir
           ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/h100_gemm.gtsir)
  add_test(NAME tile_ir_smoke COMMAND gtsim_gemm_ir ${H100_ARCH_CONFIG}
           --memory-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/memconf/full-hbm.yaml
           ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/copy.gtsir)
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
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME tile_ir_persistent_parity
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_manual_gemm.py
              --oracle $<TARGET_FILE:tile_ir_parity_tests>)
    add_test(NAME workload_profile_boundary
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_workload_profile.py)
    add_test(NAME v0_1_0_regression
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_reference.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR} --verify)
  endif()
  if(GTSIM_TVM_PYTHON)
    add_test(NAME tvm_tensor_placement
      COMMAND ${GTSIM_TVM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_placement.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME tvm_gemm_frontend
      COMMAND ${GTSIM_TVM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_gemm.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME tvm_auto_schedule
      COMMAND ${GTSIM_TVM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_auto_schedule.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME tvm_memory_search
      COMMAND ${GTSIM_TVM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_memory_search.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()

install(TARGETS gtsim_core gtsim_tile_ir gtsim_gemm gtsim_gemm_ir gtsim_fa3
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/gtsim)
install(DIRECTORY 3rdparty/glaze/include/glaze DESTINATION include)
