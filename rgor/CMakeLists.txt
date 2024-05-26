cmake_minimum_required(VERSION 3.16)
project(rgor_sys)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(OpenCV_DIR D:/opencv/build)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#find_package(Eigen3 REQUIRED CONFIG)
#find_package(Boost REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependency.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/proto PROTO_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common COMMON_SRC)


add_library(rgor_sys SHARED ${SRC} ${PROTO_SRC} ${COMMON_SRC})
target_link_libraries(rgor_sys PRIVATE
        protobuf::libprotobuf core solver_eigen types_slam3d -pthread
)
target_link_libraries(rgor_sys PUBLIC
        Eigen3::Eigen Boost::circular_buffer)
target_include_directories(rgor_sys PUBLIC ${nanoflann_SOURCE_DIR}/include ${stduuid_SOURCE_DIR}/include ${stduuid_SOURCE_DIR}/)


# add_executable(demo main.cpp)
# target_link_libraries(demo PRIVATE rgor_sys -pthread)
# message(STATUS "g2o_BINARY_DIR: ${g2o_BINARY_DIR}")
# target_link_directories(rgor_sys PRIVATE ${g2o_BINARY_DIR} ${g2o_SOURCE_DIR})

#target_link_libraries(demo PRIVATE  rgor_sys ${OpenCV_LIBS}  Eigen3::Eigen Boost::circular_buffer)
#target_include_directories(demo PRIVATE ${g2o_BINARY_DIR} ${g2o_SOURCE_DIR})
