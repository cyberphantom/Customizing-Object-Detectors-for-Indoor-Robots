#!/usr/bin/env python
'''
Copyright (C) 2018 Saif Alabachi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import print_function
import rospy
import cv2
from sensor_msgs.msg import Image
from ardrone_autonomy.msg import Navdata
from cv_bridge import CvBridge, CvBridgeError
from ucf_ardrone_ros.msg import BOX, horizontalXY, filters
from controlAgent import controller
from commandsAgent import command
#from detect_custom import DetectCustom


class drone_IO:

  def __init__(self):
    self.bridge = CvBridge()
    self.controller = controller()
    self.command = command()
    self.centroid = None
    self.move = [0, 0, 0, 0]
    self.frame = None
    self.image_pub = rospy.Publisher("/ucf_image_raw", Image)

  # Define Subscribers
  def drone_subscriber(self):
      rospy.Subscriber("/ardrone/image_raw",Image, self.callback)
      rospy.Subscriber("/ardrone/navdata", Navdata, self.ReceiveNavdata)
      #rospy.Subscriber("/box", BOX, self.boxCanvas)
      #rospy.Subscriber("/ardrone/predictedPose", filter_state, self.filter_state_callback)
      #rospy.Subscriber("/horizontalxy", horizontalXY, self.horizontalmove_callback)


  def ReceiveNavdata(self, navdata):
      self.altd = navdata.altd


  def horizontalmove_callback(self, mov):
      self.move = mov.coor


  def filter_state_callback(self, state):
      self.position = [state.x, state.y,state.z, state.yaw]


  def callback(self, data):
    try:
      self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      #Processing
      self.drone_publisher()
    except CvBridgeError as e:
      print(e)


  # Define Publishers and publish
  def drone_publisher(self):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.frame, "bgr8"))


if __name__ == '__main__':
  dio = drone_IO()
  rospy.init_node('ucf_drone', anonymous=True)
  try:
    dio.drone_subscriber()
    rospy.spin()


  except CvBridgeError as e:
    print(e)

  except KeyboardInterrupt:
    print("Shutting down")

  except rospy.ROSInterruptException:
    pass

