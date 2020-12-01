cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME BPFFNN)
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)

file(GLOB source_files
	"src/*.cpp"
)

set(SOURCES ${source_files})

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_link_libraries(${PROJECT_NAME})