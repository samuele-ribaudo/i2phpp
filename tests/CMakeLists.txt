file(GLOB TEST_EXECUTABLES "*.cpp")
foreach (file ${TEST_EXECUTABLES})
    # get the filename of the test without path
    cmake_path(GET file FILENAME name)
    # remove .cpp and use resulting string as executable name
    string(REPLACE ".cpp" "" exec ${name})
    # add the test as executable and link against test library
    add_executable(${exec} ${file})
    target_link_libraries(${exec} PRIVATE ${lib_name} gtest gmock gtest_main MPI::MPI_CXX)
endforeach ()