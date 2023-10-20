project(A2)

add_library(${PROJECT_NAME} "")

target_sources(A2 
PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/src/A2.cpp

PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include/A2.h
)
target_include_directories(A2 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

                        
                        
                        
                       
                        
                        
                        
