file(GLOB SOURCE_FILES "*.cpp")

foreach (source_file ${SOURCE_FILES})
    get_filename_component(file_name ${source_file} NAME)
    string(REPLACE ".cpp" "" exec ${file_name})
    add_executable(${exec} ${source_file})
    target_link_libraries(${exec} PUBLIC ${lib_name} MPI::MPI_CXX)
endforeach (source_file ${SOURCE_FILES})