include(FetchContent)

message(STATUS "Find package: Boost")
find_package(Boost COMPONENTS circular_buffer)
if (NOT Boost_FOUND)
    message(STATUS "Boost not found, fetching")
        set(BOOST_GIT_TAG boost-1.85.0)
        FetchContent_Declare(
                Boost_throw_exception
                GIT_REPOSITORY https://github.com/boostorg/throw_exception.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_type_traits
                GIT_REPOSITORY https://github.com/boostorg/type_traits.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_preprocessor
                GIT_REPOSITORY https://github.com/boostorg/preprocessor.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_static_assert
                GIT_REPOSITORY https://github.com/boostorg/static_assert.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_move
                GIT_REPOSITORY https://github.com/boostorg/move.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_core
                GIT_REPOSITORY https://github.com/boostorg/core.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_config
                GIT_REPOSITORY https://github.com/boostorg/config.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_concept_check
                GIT_REPOSITORY https://github.com/boostorg/concept_check.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_assert
                GIT_REPOSITORY https://github.com/boostorg/assert.git
                GIT_TAG ${BOOST_GIT_TAG}
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_Declare(
                Boost_circular_buffer
                GIT_REPOSITORY https://github.com/boostorg/circular_buffer.git
                GIT_TAG boost-1.85.0
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(Boost_throw_exception Boost_type_traits Boost_preprocessor Boost_static_assert Boost_move Boost_core Boost_config Boost_concept_check Boost_assert Boost_circular_buffer)
endif()


message(STATUS "Find package: Eigen")
find_package(Eigen3 CONFIG)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen not found, fetching")
        FetchContent_Declare(
                Eigen
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG 3.4.0
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        set(BUILD_TESTING OFF)
        set(EIGEN_BUILD_TESTING OFF)
        set(EIGEN_MPL2_ONLY ON)
        set(EIGEN_BUILD_PKGCONFIG OFF)
        set(EIGEN_BUILD_DOC OFF)
        FetchContent_MakeAvailable(Eigen)
endif()

message(STATUS "Find package: g2o")
#find_package(g2o CONFIG)
if (NOT g2o_FOUND)
    message(STATUS "g2o not found, fetching")
    FetchContent_Declare(
            g2o
            GIT_REPOSITORY https://github.com/RainerKuemmerle/g2o.git
            GIT_TAG 20230223_git
        #     GIT_SHALLOW TRUE
        #     GIT_PROGRESS TRUE
    )
#     set(G2O_BUILD_EXAMPLES OFF)
#    set(BUILD_SHARED_LIBS OFF)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    FetchContent_MakeAvailable(g2o)
#    set(g2o_DIR ${g2o_BINARY_DIR}/generated)
#    find_package(g2o REQUIRED)
endif()

message(STATUS "Find package: OpenCV")
find_package(OpenCV 4)
if (NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found, fetching")
        FetchContent_Declare(
                opencv
                GIT_REPOSITORY https://github.com/opencv/opencv.git
                GIT_TAG        4.9.0
                # GIT_SHALLOW TRUE
                # GIT_PROGRESS TRUE
        )
        FetchContent_GetProperties(opencv)
        if (NOT opencv_POPULATED)
                FetchContent_Populate(opencv)
        endif ()
        FetchContent_MakeAvailable(opencv)
        set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR})
        include_directories(${OpenCV_INCLUDE_DIRS})
        find_package(OpenCV REQUIRED)
endif()

message(STATUS "Fetching nanoflann")
FetchContent_Declare(
        nanoflann
        GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
        GIT_TAG        v1.5.5
        # GIT_SHALLOW TRUE
        # GIT_PROGRESS TRUE
)
set(MASTER_PROJECT_HAS_TARGET_UNINSTALL ON)
set(NANOFLANN_BUILD_EXAMPLES OFF)
FetchContent_MakeAvailable(nanoflann)
include_directories(${nanoflann_SOURCE_DIR}/include)

message(STATUS "Fetching protobuf")
# FetchContent_Declare(
#     Protobuf
#     GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
#     # Needs to match the Protobuf version that libprotobuf-mutator is written for, roughly.
#     GIT_TAG v22.3
#     GIT_SHALLOW ON

#     # libprotobuf-mutator will need to be able to find this at configuration
#     # time.
#     OVERRIDE_FIND_PACKAGE
# )

# set(protobuf_BUILD_TESTS OFF)
# set(protobuf_BUILD_SHARED_LIBS OFF)
# # libprotobuf-mutator relies on older module support.
# set(protobuf_MODULE_COMPATIBLE ON)

# FetchContent_MakeAvailable(Protobuf)

# find_package(Protobuf CONFIG REQUIRED)



FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG        v21.4
        # GIT_SHALLOW    TRUE
        # GIT_PROGRESS TRUE
          SOURCE_SUBDIR  cmake
  FIND_PACKAGE_ARGS NAMES protobuf
)
set(protobuf_BUILD_TESTS OFF)
set(gmock_build_tests OFF)
set(protobuf_BUILD_SHARED_LIBS OFF)
set(protobuf_INSTALL OFF)
set(protobuf_WITH_ZLIB OFF)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
FetchContent_MakeAvailable(protobuf)
set(protobuf_DIR "${protobuf_BINARY_DIR}/lib/cmake/protobuf")
find_package(Protobuf REQUIRED)

message(STATUS "Fetching stduuid")
FetchContent_Declare(
            stduuid
            GIT_REPOSITORY https://github.com/mariusbancila/stduuid.git
            GIT_TAG v1.2.3
        #     GIT_SHALLOW TRUE
        #     GIT_PROGRESS TRUE
    )
#     set(UUID_SYSTEM_GENERATOR ON)
    FetchContent_MakeAvailable(stduuid)
    include_directories(${stduuid_SOURCE_DIR})
include_directories(${stduuid_SOURCE_DIR}/include)


#FetchContent_Declare(
#        dlib
#        GIT_REPOSITORY https://github.com/davisking/dlib.git
#        GIT_TAG        v19.24.4
#)
#FetchContent_MakeAvailable(dlib)

#spdlog