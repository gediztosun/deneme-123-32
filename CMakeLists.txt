cmake_minimum_required(VERSION 3.28)
project(untitled)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_PREFIX_PATH "/home/gediz/Qt/5.15.16/gcc_64/lib/cmake")

find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        DataVisualization
        REQUIRED)

find_package(Boost REQUIRED COMPONENTS serialization)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(untitled main.cpp
        scatterdatamodifier.cpp
        scatterdatamodifier.h)
target_link_libraries(untitled
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::DataVisualization
        ${Boost_LIBRARIES}
        Eigen3::Eigen
)

