include_directories(.)
file(GLOB GAMES_SRC "*/*.cc")

add_library(czf.czf_env OBJECT game.cc ${GAMES_SRC})

pybind11_add_module(czf_env
  module.cc
  $<TARGET_OBJECTS:czf.czf_env>
)
install(TARGETS czf_env LIBRARY DESTINATION czf/env/)