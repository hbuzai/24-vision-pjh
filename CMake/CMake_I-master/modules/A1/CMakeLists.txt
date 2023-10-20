project(A1)

add_library(${PROJECT_NAME} "")

target_sources(A1 
PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/src/A11.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/A12.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/A13.cpp
PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include/A1.h
)
target_include_directories(A1 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

