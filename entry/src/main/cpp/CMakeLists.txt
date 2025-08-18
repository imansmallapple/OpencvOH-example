# the minimum version of CMake
cmake_minimum_required(VERSION 3.5.0)
project(Opencv)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(OPENCV_LIB_PATH ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH})
set(OPENCV_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH})
link_directories(${OPENCV_LIB_DIR})
include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

add_library(entry SHARED
            common/common.cpp
            image_basic/image2Gray.cpp
            napi_init.cpp)

target_link_libraries(entry PUBLIC
    libace_napi.z.so
    librawfile.z.so
    libhilog_ndk.z.so
    ${OPENCV_LIB_PATH}/libopencv_core.so
    opencv_imgcodecs
    opencv_imgproc
    opencv_photo
    opencv_videoio
    opencv_flann
    opencv_features2d
    opencv_calib3d
    opencv_video
    opencv_ml
    opencv_highgui
    opencv_stitching
    opencv_dnn
    opencv_objdetect
    opencv_gapi)
