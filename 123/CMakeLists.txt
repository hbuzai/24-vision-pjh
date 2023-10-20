cmake_minimum_required(VERSION 3.0)
project(TEST)
find_package(OpenCV)
#link_directories(include )

add_executable(test main.cpp)
target_link_libraries(test ${OpenCV_LIBS})


#target_include_directories(test PUBLIC include )