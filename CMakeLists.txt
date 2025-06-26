find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)
include(UseCython)

cython_transpile(pytential/qbx/target_specific/impl.pyx LANGUAGE C OUTPUT_VARIABLE pytential_c)

python_add_library(impl
    MODULE
    "${pytential_c}"
    pytential/qbx/target_specific/helmholtz_utils.c
    WITH_SOABI)

target_compile_options(impl PRIVATE -Wall -Ofast -fopenmp)
target_link_options(impl PRIVATE -fopenmp)
target_include_directories(impl PRIVATE ${CMAKE_SOURCE_DIR}/pytential/qbx/target_specific)

install(TARGETS impl DESTINATION pytential/qbx/target_specific)
