cmake_minimum_required(VERSION 3.10)
project(miniCodec LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 17)

enable_language(C)
enable_language(CUDA)

# Add debug symbols and AVX flags
add_compile_options(-g -mavx)

find_package(CUDA REQUIRED)
find_package(JPEG REQUIRED)

include_directories(${JPEG_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/tests)

# Glob source files
file(GLOB_RECURSE SRC_FILES "${SRC_DIR}/*.c" "${SRC_DIR}/*.cu")
file(GLOB_RECURSE TEST_FILES "${TEST_DIR}/*.c" "${TEST_DIR}/*.cu")

message(STATUS "Source files found:")
foreach(file ${SRC_FILES})
    message(STATUS "  ${file}")
endforeach()

# Add source files to a library
add_library(miniCodec_lib ${SRC_FILES})

# Set include directories for the library
target_include_directories(miniCodec_lib PUBLIC ${SRC_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests)

# Link libraries
target_link_libraries(miniCodec_lib PUBLIC ${JPEG_LIBRARIES})
target_link_libraries(miniCodec_lib PUBLIC ${CUDA_LIBRARIES})

# Add debug-specific flags to the targets
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

    # Create executable for each test file
    add_executable(${TEST_NAME} ${TEST_FILE})

    # Link libraries to the test executable
    target_link_libraries(${TEST_NAME} miniCodec_lib ${JPEG_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_options(${TEST_NAME} PRIVATE -g)  # Ensure debug flags are added
endforeach()
