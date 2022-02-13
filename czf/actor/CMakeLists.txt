find_package(Torch REQUIRED)
find_package(
  Python3
  COMPONENTS Development
  REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(alphazero_worker)
add_subdirectory(muzero_worker)
add_subdirectory(muzero_worker_gumbel)
