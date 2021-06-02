include_directories(.)
file(GLOB GAMES_SRC "*/*.cc")

add_library(tests.czf_env OBJECT test_czf_env.cc ${GAMES_SRC})
target_link_libraries(tests.czf_env PRIVATE Catch2::Catch2)

add_executable(test_czf_env $<TARGET_OBJECTS:tests.czf_env>
                            $<TARGET_OBJECTS:czf.czf_env>)

add_test(NAME czf-env-tests COMMAND test_czf_env)