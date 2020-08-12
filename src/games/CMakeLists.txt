file(GLOB CZF_GAMES_SRC "*.cc")
add_library(games OBJECT ${CZF_GAMES_SRC})
target_include_directories(games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})