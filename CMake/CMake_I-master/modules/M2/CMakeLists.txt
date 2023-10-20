project(M2)

add_library(${PROJECT_NAME} "")
find_package(OpenCV)
target_sources(M2 
PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/src/M2.cpp

PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include/M2.h
    /home/ladistra/24-vision-pjh/CMake/CMake_I-master/modules/A1/include/A1.h
    /home/ladistra/24-vision-pjh/CMake/CMake_I-master/modules/A2/include/A2.h
)
target_include_directories(M2 
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}/include
    /home/ladistra/24-vision-pjh/CMake/CMake_I-master/modules/A1/include
    /home/ladistra/24-vision-pjh/CMake/CMake_I-master/modules/A2/include
    )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

