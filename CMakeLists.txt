cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(rk3588_demo VERSION 0.0.1 LANGUAGES CXX)

message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)

find_package(OpenCV REQUIRED)

include_directories(
    ${RKNN_API_INCLUDE_PATH}
    ${OpenCV_INCLUDE_DIRS}
    ${RGA_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/draw
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/process
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rkmedia
    ${CMAKE_CURRENT_SOURCE_DIR}/src/task
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)


add_library(nn_process SHARED
            src/process/preprocess.cpp
            src/process/postprocess.cpp
)
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

add_library(draw_lib SHARED src/draw/cv_draw.cpp)
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

add_library(rknn_engine SHARED src/engine/rknn_engine.cpp)
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
    draw_lib
)

add_library(yolox_lib SHARED src/task/yolox_custom.cpp)
target_link_libraries(yolox_lib
    rknn_engine
    nn_process
)


add_executable(main
        src/main.cpp
        src/task/yolox_custom.cpp
)

target_link_libraries(main
        draw_lib
        yolox_lib
)
