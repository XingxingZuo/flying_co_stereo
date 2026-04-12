cmake_minimum_required(VERSION 3.5)
project(depth_align)

#set(cv_bridge_DIR ~/tools_ws/install/share/cv_bridge/cmake)
# set(OpenCV_DIR /usr/local/share/OpenCV)

find_package(OpenCV QUIET)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED) #COMPONENTS common io visualization filters
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(yaml-cpp REQUIRED)
find_package(uav_config REQUIRED)
find_package(catkin REQUIRED COMPONENTS
#  ov_core
  roscpp rospy cv_bridge
  std_msgs visualization_msgs geometry_msgs nav_msgs tf rosbag pcl_ros sensor_msgs image_transport
  message_generation
  #swarm_msgs
)

add_definitions(${PCL_DEFINITIONS})
set(ENABLE_BACKWARD true)
set(REAL_ENV true) # in real world, set to true; in simulator, set to false

if (REAL_ENV)
    add_definitions(-D real_data)
else()
    add_definitions(-D sim_data)
endif ()

if (ENABLE_BACKWARD)
    add_definitions(-D USE_BACKWARD)
endif ()

# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
### C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES relative_pose_backend
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)


###########
## Build ##
###########
include_directories(
        include
        src
        ../../devel/include
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${uav_config_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
)

list(APPEND thirdparty_libraries
       ${Boost_LIBRARIES}
       ${OpenCV_LIBRARIES}
       ${catkin_LIBRARIES}
       )

# use Ceres to global align DepthAnything and sparse landmarks
add_executable(depth_align_node src/depth_align_node.cpp)
target_link_libraries(depth_align_node ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${CERES_LIBRARIES} stdc++fs dw)
