find_package(Doxygen)
if(DOXYGEN_FOUND)
  # Doxygen
  set(DOXYGEN_GENERATE_XML YES)
  set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
  doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/czf)

  # Sphinx
  find_package(Sphinx REQUIRED)
  set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
  set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
  set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

  file(GLOB_RECURSE SPHINX_DOCS "${SPHINX_SOURCE}/*.rst")
  add_custom_command(
    OUTPUT ${SPHINX_INDEX_FILE}
    COMMAND
      ${SPHINX_EXECUTABLE} -b html
      -Dbreathe_projects.czf=${DOXYGEN_OUTPUT_DIRECTORY}/xml ${SPHINX_SOURCE}
      ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS doxygen ${SPHINX_DOCS}
    MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
    COMMENT "Generating documentation with Sphinx")
  add_custom_target(
    sphinx
    DEPENDS ${SPHINX_INDEX_FILE}
    COMMENT "Sphinx documentation target")
endif(DOXYGEN_FOUND)
