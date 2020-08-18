cmake_minimum_required(VERSION 2.8)

project("dataRRANSAC")
set(${PROJECT_NAME}_MAJOR 1)
set(${PROJECT_NAME}_MINOR 0)

set(CMAKE_CXX_FLAGS "${CMAKE_CSS_FLAGS} -std=c++11")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -pedantic-errors -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE:STRING=RelWithDebInfo)

find_package(OpenCV REQUIRED)

# Include headers
include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../imfw
)

add_executable (
    ${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/rransacFrameN.cpp
)

target_link_libraries (
    ${PROJECT_NAME} ${OpenCV_LIBS}
)
