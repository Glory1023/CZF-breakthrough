#! /bin/sh
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SOURCE_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"

rm -rf "$SCRIPT_DIR/bin"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

protoc -I="$SOURCE_DIR" \
  --cpp_out="$SOURCE_DIR" \
  --python_out="$SOURCE_DIR/python" \
  "$SOURCE_DIR/czf.proto"

cmake -D "CMAKE_INSTALL_PREFIX=$SCRIPT_DIR" -S "$SOURCE_DIR" -B "$BUILD_DIR"
make -C "$BUILD_DIR" -j