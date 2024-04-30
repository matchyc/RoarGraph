set(CMAKE_CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME}
	gtest
	gtest_main
	# mockcpp
	pthread
	)

	# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable

add_executable(test_build_roargraph test_build_roargraph.cpp)
target_link_libraries(test_build_roargraph ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)


add_executable(test_search_roargraph test_search_roargraph.cpp)
target_link_libraries(test_search_roargraph ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_only_build_bipartite test_only_build_bipartite.cpp)
target_link_libraries(test_only_build_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)

add_executable(test_search_bipartite test_search_bipartite.cpp)
target_link_libraries(test_search_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)