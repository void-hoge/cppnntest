project(nntest)

set(CMAKE_CXX_STANDARD 20)
set("CMAKE_CXX_FLAGS" "-O3 -mtune=native -march=native")

add_executable(nntest main.cpp model.cpp layer.cpp activation.cpp)
