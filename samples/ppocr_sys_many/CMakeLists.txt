cmake_minimum_required(VERSION 3.10.2)

project(cvi_sample_ppocr_sys_many C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Ofast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

if(NOT DEFINED TPU_SDK_PATH)
  message(FATAL_ERROR "Please set TPU_SDK_PATH to point to the TPU_SDK installation")
endif()
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

if(NOT DEFINED OPENCV_PATH)
  message(FATAL_ERROR "Please set OPENCV_PATH to point to the opencvn installation")
endif()
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

set(CVI_LIBS ${CVI_LIBS} cviruntime cvikernel)
if(NOT CMAKE_CROSSCOMPILING)
  set(CVI_LIBS ${CVI_LIBS} cvicmodel)
endif()

set(OPENCV_LIBS ${OPENCV_LIBS} opencv_core opencv_imgcodecs opencv_imgproc)
if(NOT CMAKE_CROSSCOMPILING)
  set(OPENCV_LIBS ${OPENCV_LIBS} opencv_highgui)
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread z)

add_executable(cvi_sample_ppocr_sys_many ppocr_sys_many.cpp)
target_link_libraries(cvi_sample_ppocr_sys_many
  ${CVI_LIBS}
  ${OPENCV_LIBS}
  ${EXTRA_LIBS})
install(TARGETS cvi_sample_ppocr_sys_many cvi_sample_ppocr_sys_many DESTINATION bin)
