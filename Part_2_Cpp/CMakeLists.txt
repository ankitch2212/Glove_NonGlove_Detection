cmake_minimum_required(VERSION 3.16)
project(glove_detector_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV -------------------------------------------------------------
# The prebuilt Windows package ships OpenCVConfig.cmake under <opencv>/build,
# so find_package works once OpenCV_DIR points there, e.g.:
#   -DOpenCV_DIR=C:/Users/ankit/cpp_deps/opencv/build
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

# --- ONNX Runtime ---------------------------------------------------------
# The plain win-x64 release zip ships NO CMake config package, so we find it
# by hand. Pass its extracted root, e.g.:
#   -DONNXRUNTIME_ROOT=C:/Users/ankit/cpp_deps/onnxruntime-win-x64-1.30.0
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of the extracted onnxruntime-win-x64-<version> release")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT}/include $ENV{ONNXRUNTIME_ROOT}/include)
find_library(ORT_LIB onnxruntime
          HINTS ${ONNXRUNTIME_ROOT}/lib $ENV{ONNXRUNTIME_ROOT}/lib)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found. Configure with "
    "-DONNXRUNTIME_ROOT=<path to extracted onnxruntime-win-x64-*> "
    "(the folder that contains include/ and lib/).")
endif()

add_executable(glove_detector
  src/main.cpp
  src/detector.cpp
  src/utils.cpp
  src/config.cpp)

target_include_directories(glove_detector PRIVATE include ${ORT_INCLUDE_DIR})
target_link_libraries(glove_detector PRIVATE ${OpenCV_LIBS} ${ORT_LIB})

# Copy the runtime DLLs next to the exe so it launches without editing PATH.
if(WIN32)
  find_file(ORT_DLL onnxruntime.dll HINTS ${ONNXRUNTIME_ROOT}/lib)
  if(ORT_DLL)
    add_custom_command(TARGET glove_detector POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" $<TARGET_FILE_DIR:glove_detector>)
  endif()

  # OpenCV's DLL lives under <OpenCV_DIR>/../bin (vc16/bin next to the build/ config dir).
  file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/vc16/bin/opencv_world*.dll")
  foreach(dll ${OPENCV_DLLS})
    add_custom_command(TARGET glove_detector POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" $<TARGET_FILE_DIR:glove_detector>)
  endforeach()
endif()
