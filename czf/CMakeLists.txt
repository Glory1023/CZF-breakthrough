add_library(czf OBJECT czf.cc czf.h)
set(CZF_OBJECTS $<TARGET_OBJECTS:czf>)
# target_include_directories(czf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(dummy_agent dummy_agent.cc ${CZF_OBJECTS})
set_target_properties(dummy_agent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)