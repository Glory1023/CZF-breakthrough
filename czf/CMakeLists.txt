add_subdirectory(actor)

find_program(PROTOC_BIN protoc DOC "Protocl Buffers Compiler")
add_custom_target(proto)
add_custom_command(
  TARGET proto
  COMMAND ${PROTOC_BIN} ARGS -I proto --python_out pb proto/czf.proto
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
