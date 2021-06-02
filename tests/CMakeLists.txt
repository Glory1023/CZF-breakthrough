include_directories(..)
add_subdirectory(env)

# test module
find_package(Python3)
add_test(
  NAME czf-env-pybind-tests
  COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
)