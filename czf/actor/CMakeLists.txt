file(GLOB CZF_PYTHON_SRC "*.py")
file(COPY ${CZF_PYTHON_SRC} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

pybind11_add_module(pyczf pyczf.cc ${CZF_OBJECTS})
set_target_properties(pyczf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
target_include_directories(pyczf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})