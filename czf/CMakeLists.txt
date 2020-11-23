find_package(Protobuf REQUIRED)
include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/czf.proto)
protobuf_generate_python(PROTO_PY proto/czf.proto)
add_library(czf.proto OBJECT ${PROTO_SRCS})

add_subdirectory(actor)
