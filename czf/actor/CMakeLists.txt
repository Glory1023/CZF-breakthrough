include_directories(.)
add_subdirectory(worker)

# pybind11 module
pybind11_add_module(worker module.cc $<TARGET_OBJECTS:czf.workers>)
install(TARGETS worker LIBRARY DESTINATION czf/actor)

# test module add_test(NAME pybind-tests COMMAND python -m pytest
# ${PROJECT_SOURCE_DIR}/tests)
