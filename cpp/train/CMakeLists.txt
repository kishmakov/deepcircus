add_executable(graph_vis main.cpp)
target_link_libraries(graph_vis PRIVATE tools)
set_source_files_properties(main.cpp PROPERTIES COMPILE_OPTIONS "-UNDEBUG")
