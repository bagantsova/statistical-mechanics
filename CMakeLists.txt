cmake_minimum_required(VERSION 3.16)
project(statistical-mechanics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(OpenMP)

add_executable(md_lennard_jones
  md-lennard-jones/src/main.cpp
  md-lennard-jones/src/system.cpp)

add_executable(ising_2d ising/src/main.cpp)

foreach(target md_lennard_jones ising_2d)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found - ${target} will run single-threaded")
  endif()
endforeach()

enable_testing()

# Both experiments are checked against a known answer rather than inspected.
# These run the check and fail the build's test step if it stops holding.
add_test(NAME md_conserves_energy
         COMMAND md_lennard_jones --particles 108 --box 6.0 --steps 500
                                  --dump-every 0 --output ${CMAKE_CURRENT_BINARY_DIR}/test-md)
add_test(NAME ising_recovers_onsager_tc
         COMMAND ising_2d --sizes 8,16 --points 13 --equilibrate 500 --measure 2000
                          --output ${CMAKE_CURRENT_BINARY_DIR}/test-ising)
# Assert on the numbers, not just the exit code: the MD drift must stay below
# 1e-3, and the Ising crossing must land within 1 % of the exact T_c.
set_tests_properties(md_conserves_energy PROPERTIES
                     PASS_REGULAR_EXPRESSION "max relative energy drift: 0\\.000")
set_tests_properties(ising_recovers_onsager_tc PROPERTIES
                     PASS_REGULAR_EXPRESSION "relative error +0\\.[0-9]+ %")
