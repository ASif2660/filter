cmake_minimum_required(VERSION 3.8)

project(packageWithCuda CXX CUDA)
set (CMAKE_CXX_STANDARD 11)

#findpackages
find_package(OpenCV REQUIRED)
#find_package(Eigen3 REQUIRED)

set(CUDAS_INCLUDE_DIR "/usr/local/cuda/include")


# help CLion engine with CUDA headers
include_directories(${CUDAS_INCLUDE_DIR}) #CUDAS to avoid any internal clash
# help CLion engine with eigen headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen/include/eigen3 )

#creating libraries
add_library(myLib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cuda-support/src/filter.cu
                                   ${CMAKE_CURRENT_SOURCE_DIR}/cuda-support/include/filter.cuh) #this includes other modules into the main project #the control goes to this directory now


#appending targets
target_link_libraries(myLib ${OpenCV_LIBS})
#here you can have more add_libraries if you want to.

#creating exec
add_executable(filter main.cpp)


#final targets
target_link_libraries(filter myLib)



## Install the executable to <install_directory>/bin
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
## Install libraries (if any) to <install_directory>/bin
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

#copying the image, this is ony for test
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/support/image/x.jpg
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)


add_custom_target( testdetector ALL
        ${CMAKE_COMMAND} -E cmake_echo_color --cyan "target is compiled"
        COMMENT "is located at build_directory/bin? " )


add_dependencies( testdetector filter )



