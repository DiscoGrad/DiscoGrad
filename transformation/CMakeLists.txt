cmake_minimum_required()
project(DiscoGradTransformations)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_BUILD_TYPE "Debug")

find_package(LLVM REQUIRED CONFIG)
execute_process(COMMAND llvm-config --libs all OUTPUT_VARIABLE LLVM_LIBRARIES)
string(STRIP ${LLVM_LIBRARIES} LLVM_LIBRARIES)
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM libraries: ${LLVM_LIBRARIES}")

add_compile_options(-Wno-deprecated-enum-enum-conversion)

option(SMOOTH_LOOPS "Smooth loops" OFF)
if (SMOOTH_LOOPS)
add_compile_definitions(SMOOTH_LOOPS=1)
endif()

# Add targets and link LLVM and clang libraries
set(TARGET_LIST normalize;smooth_dgsi;smooth_dgo)
foreach (item ${TARGET_LIST})
    add_executable(${item} ${item}.cpp)
    target_link_libraries(${item} ${LLVM_LIBRARIES})
    # for systems where llvm is linked with a single lib
    # e.g. Fedora, Arch Linux
    find_library(LLVM_LIB clang-cpp)
    if(LLVM_LIB)
      target_link_libraries(${item} clang-cpp)
    else()
      target_link_libraries(${item}
              clangFrontend
              clangSerialization
              clangDriver
              clangParse
              clangSema
              clangAnalysis
              clangAST
              clangBasic
              clangEdit
              clangLex
              clangTooling
              clangRewrite
              )
    endif()
endforeach (item)
