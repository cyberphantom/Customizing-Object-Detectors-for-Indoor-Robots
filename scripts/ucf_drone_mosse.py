#!/usr/bin/env python

'''
Copyright (C) 2018 Saif Alabachi
based on MOSSE object detection tracker

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
import time
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from ucf_ardrone_ros.msg import BOX, horizontalXY, filters
from tum_ardrone.msg import filter_state
from cv_bridge import CvBridge, CvBridgeError
from tracker_agent.mosseTracker import MOSSE
from control_agent.controlAgent import controller
from control_agent.command_agent.commandsAgent import command
import filters_agent.filtersAgent as FL




class image_converter:

  def __init__(self):
    self.drag_rect = None
    self.controller = controller()
    self.command = command()
    self.image_pub = rospy.Publisher("/mosse",Image, queue_size=10)
    self.bridge = CvBridge()
    self.box_sub = rospy.Subscriber("/box", BOX, self.boxCanvas)
    self.image_sub = rospy.Subscriber("/ardrone/image_raw", Image, self.callback)
    self.filter_state = rospy.Subscriber("/ardrone/predictedPose", filter_state,
                                         self.filter_state_callback)
    rospy.Subscriber("/horizontalxy", horizontalXY, self.movement_callback)
    rospy.Subscriber("/ardrone/navdata", Navdata, self.ReceiveNavdata)
    rospy.Subscriber("/filters", filters, self.filters_calback)
    self.centroid = None
    self.filtered_frame = None
    self.done = False
    self.trackers = []
    self.x = 0
    self.y = 0
    self.prev_time = 0

  def filters_calback(self, fl):
      if fl is not None:
          frame = self.frame
          self.filtered_frame = FL.apply_filters(frame, fl)

  def movement_callback(self, hxy):
      self.x = hxy.x
      self.y = hxy.y

  def ReceiveNavdata(self, navdata):
      self.altd = navdata.altd

  def filter_state_callback(self, state):
      self.position = [state.x, state.y,state.z, state.yaw]

  def boxCanvas(self, data):
      if data.data[0] == -1 and data.data[1] == -1 and data.data[2] == -1 \
              and data.data[3] == -1:
        self.trackers = []
        self.command.hover()
      else:
          if data.data:
            self.drag_rect = tuple(data.data)
            self.onrect(self.drag_rect)


  def callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.frame = cv_image
      if not self.done:
        self.done = True

    except CvBridgeError as e:
      print(e)

    frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    if self.filtered_frame is not None:
        vis = self.filtered_frame
    else:
        vis = self.frame.copy()
    if self.trackers:
        for tracker in self.trackers:
            ticks = time.time()
            delta_time = ticks - self.prev_time
            folder = 'frames/'
            if delta_time >= 0.5:
                cv2.imwrite(os.path.join(folder, str(ticks).split(".")[0]) +
                            '.png', vis[int(tracker.pos[1]) -
                                       int(tracker.size[1]/2):int(tracker.pos[1]) +
                                                              int(tracker.size[1]/2),
                                   int(tracker.pos[0]) -
                                   int(tracker.size[0]/2):int(tracker.pos[0]) +
                                                          int(tracker.size[0]/2)])
                self.prev_time = ticks
            tracker.update(frame_gray)
            self.centroid = tracker.pos
            self.controller.correct_position(self.x, self.y, self.centroid, self.altd)
    else:
         self.centroid = None


    for tracker in self.trackers:
      tracker.draw_state(vis)

    self.draw()
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
    except CvBridgeError as e:
      print(e)

  def draw(self):
      if not self.drag_rect:
          return False
      x0, y0, x1, y1 = self.drag_rect
      return True

  @property
  def dragging(self):
      return self.drag_rect is not None

  def rect(self, data):
    self.trackers.append(data)

  def onrect(self, rect): #saif
    if self.filtered_frame is not None:
        frame_gray = cv2.cvtColor(self.filtered_frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    tracker = MOSSE(frame_gray, rect)
    self.trackers.append(tracker)


def main():
  rospy.init_node('ucf_drone', anonymous=True)
  image_converter()
  try:
    rospy.spin()
    image_converter()

  except KeyboardInterrupt:
    print("Shutting down")

  except rospy.ROSInterruptException:
        pass
  cv2.destroyAllWindows()

if __name__ == '__main__':
    print(os.path.join(os.getcwd(),"frames", "saif"))
    main()


