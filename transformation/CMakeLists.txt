cmake_minimum_required()
project(DiscoGradTransformations)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_BUILD_TYPE "Debug")

find_package(LLVM REQUIRED CONFIG)
execute_process(COMMAND llvm-config --libs all OUTPUT_VARIABLE LLVM_LIBRARIES COMMAND_ERROR_IS_FATAL ANY)
string(STRIP ${LLVM_LIBRARIES} LLVM_LIBRARIES)
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM libraries: ${LLVM_LIBRARIES}")

add_compile_options(-Wno-deprecated-enum-enum-conversion)

# Add targets and link LLVM and clang libraries
set(TARGET_LIST normalize;smooth_dgo)
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
