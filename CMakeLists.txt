cmake_minimum_required(VERSION 2.8.3)
project(ucf_ardrone_ros)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  rospy_tutorials
  message_generation
  std_srvs
  tf
  camera_info_manager
  roslint
)

# Adding other required ROS packages 
find_package(OpenCV REQUIRED)


# Declaring the required messages
add_message_files(
	FILES 
	filters.msg 
	BOX.msg
	horizontalXY.msg
)



generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport roscpp rospy sensor_msgs std_msgs message_runtime tf camera_info_manager

)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable (ucf_drone_pos  src/ucf_drone_pos.cpp)
add_executable (ucf_drone_auto src/ucf_drone_auto.cpp)

add_dependencies(ucf_drone_pos
		 ucf_drone_auto
 ${PROJECT_NAME})


target_link_libraries(ucf_drone_pos
   ${catkin_LIBRARIES})
target_link_libraries(ucf_drone_auto
   ${catkin_LIBRARIES})

install(PROGRAMS		scripts/control_agent/controlAgent.py
				scripts/commands_agent/commandsAgent.py	 		 
				scripts/ucf_drone_io.py
				scripts/ucf_drone_mosse.py
				scripts/image_collector_agent.py
                		scripts/ucf_drone_object_detection.py
				scripts/myFiles/listener.py

        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)



