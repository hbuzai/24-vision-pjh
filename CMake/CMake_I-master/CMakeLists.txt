# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.0)
project(TEST)

####### 以下禁止修改 #######
add_subdirectory(common)   #
add_subdirectory(modules)  #
####### 以上禁止修改 #######

find_package(OpenCV)
aux_source_directory(modules/A1/src SRC_LIST)
aux_source_directory(modules/A2/src  SRC_LIST2)
aux_source_directory(modules/M1/src SRC_LIST3)
aux_source_directory(modules/M2/src SRC_LIST4)
include_directories(modules/A1/include modules/A2/include modules/M1/include modules/M2/include common/math/include)

# 在此处生成可执行程序
add_executable(test main.cpp ${SRC_LIST} ${SRC_LIST1} ${SRC_LIST3} ${SRC_LIST4})


# 在此处链接库
target_link_libraries(test 
PRIVATE
    ${OpenCV_LIBS} 
    A1 
    A2 
    M1 
    M2 
)


