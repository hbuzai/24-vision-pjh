project(M1)

add_library(${PROJECT_NAME} "")

target_sources(M1 
PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/src/M1.cpp

PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include/M1.h
    /home/ladistra/24-vision-pjh/CMake/CMake_I-master/modules/A1/include/A1.h
)
target_include_directories(M1 
PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}/include
    /home/ladistra/24-vision-pjh/CMake/CMake_I-master/modules/A1/include
    )
