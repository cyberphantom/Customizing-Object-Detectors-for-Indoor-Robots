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
import rospy, rosbag, subprocess, yaml
import cv2, time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from cv_bridge import CvBridge, CvBridgeError
from ucf_ardrone_ros.msg import BOX, horizontalXY, filters
from control_agent.controlAgent import controller
from control_agent.commands_agent.commandsAgent import command
from filters_agent.filtersAgent import apply_filters as FL
import os, csv, math

''' Using Object Detection API '''
#from object_detection_agent.research.od_detect_objects import DetectImage

''' Using Object Detection API with custom objects '''
#from object_detection_agent.research.od_detect_custom_object import DetectCustom
#from object_detection_agent.od_detect_custom_object import DetectCustom

''' Using dronet Object Detection '''
#from object_detection_agent.research.od_detect_custom_object import DetectCustom
#from object_detection_agent.dronet.dronet_detector import dronet_detect
from object_detection_agent.DUNet.dunet_detector_ssd import dronet_detect

''' Using Object Detection API with mix objects '''
#from object_detection_agent.research.od_two_models import DetectMix

class drone_IO:

  '''default object detection'''
  #di = DetectImage()

  '''Detect custom objects'''
  #di = DetectCustom()

  '''Detect dronet'''
  di = dronet_detect()

  '''Detect mix objects'''
  #di = DetectMix()

  def __init__(self):
    self.bridge = CvBridge()
    self.controller = controller()
    self.command = command()
    self.move = [0, 0, 0, 0]
    self.frame = None
    self.detected_frame = None
    self.filtered_frame = None
    self.pre_filter = None
    self.filters = None
    self.h_x = 0
    self.h_y = 0
    #self.image_pub = rospy.Publisher("/ucf_image_raw", Image)
    #self.dist = rospy.Publisher("/dist_to_obj", Twist)
    self.initTime = None
    self.a0, self.d0 = None, None
    self.a1, self.d1 = None, None
    self.a2, self.d2 = None, None
    self.mosses = []
    self.re3_bbx = None
    self.drag_rect = None
    self.twist = Twist()
    self.folder = '/home/saif/bag_files/uav2/'
    self.data_file = []
    self.start = None
    self.end = 0
    self.tick =0
    self.prev_time = None
    '''Using Onlinre Trackers'''
    #self.online_tracker = init_tracker(4)

    '''run the bag file'''


  '''Define Subscribers'''
  def drone_subscriber(self):

      rospy.init_node('ucf_drone', anonymous=True)

      # Subscribe to ardrone_autonomy
      rospy.Subscriber("/ardrone/image_raw", Image, self.callback)
      #rospy.Subscriber("/ardrone/navdata", Navdata, self.receiveNavdata_callback)

      # Listen to Filters changing
      #rospy.Subscriber("/filters", filters, self.filters_calback)

      rospy.spin()

  def filters_calback(self, fl):
      if fl is not None:
          self.filters = fl.fls


  def receiveNavdata_callback(self, navdata):
      self.altd = navdata.altd



  def callback(self, data):


    try:

      #self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.frame = self.bridge.imgmsg_to_cv2(data, "rgb8")

      '''Processing'''
      if self.filters is not None:
        self.filtered_frame = FL(self.frame, self.filters)
        frame = self.filtered_frame
      else:
        frame = self.frame.copy()

      if self.start is None:
          self.start = time.time()

      '''Call detection before tracking if requested'''

      self.before_det_t = time.time() - self.start

      #self.detected_frame, info = self.di.run_detect(frame)
      self.detected_frame, info = self.di.predictor(frame)

      self.after_det_t = time.time() - self.start

      det_time = self.after_det_t - self.before_det_t

      #print(info)

      if len(info) > 0:

          if len(info) == 1:
            #print(str(info[0].partition(':')[0]), float(info[0].partition(':')[2]))
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2]])
          elif len(info) == 2:
            #print(info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2])
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2]])
          elif len(info) == 3:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2]])
          elif len(info) == 4:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2]])
          elif len(info) == 5:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2], info[4].partition(':')[0], info[4].partition(':')[2]])
          elif len(info) == 6:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2], info[4].partition(':')[0], info[4].partition(':')[2], info[5].partition(':')[0], info[5].partition(':')[2]])
          elif len(info) == 7:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2], info[4].partition(':')[0], info[4].partition(':')[2], info[5].partition(':')[0], info[5].partition(':')[2], info[6].partition(':')[0], info[6].partition(':')[2]])
          elif len(info) == 8:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2], info[4].partition(':')[0], info[4].partition(':')[2], info[5].partition(':')[0], info[5].partition(':')[2], info[6].partition(':')[0], info[6].partition(':')[2], info[7].partition(':')[0], info[7].partition(':')[2]])
          elif len(info) == 9:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2], info[4].partition(':')[0], info[4].partition(':')[2], info[5].partition(':')[0], info[5].partition(':')[2], info[6].partition(':')[0], info[6].partition(':')[2], info[7].partition(':')[0], info[7].partition(':')[2], info[8].partition(':')[0], info[8].partition(':')[2]])
          elif len(info) == 10:
            self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), info[0].partition(':')[0], info[0].partition(':')[2], info[1].partition(':')[0], info[1].partition(':')[2], info[2].partition(':')[0], info[2].partition(':')[2], info[3].partition(':')[0], info[3].partition(':')[2], info[4].partition(':')[0], info[4].partition(':')[2], info[5].partition(':')[0], info[5].partition(':')[2], info[6].partition(':')[0], info[6].partition(':')[2], info[7].partition(':')[0], info[7].partition(':')[2], info[8].partition(':')[0], info[8].partition(':')[2], info[9].partition(':')[0], info[9].partition(':')[2]])


      else:
          self.data_file.append([truncate(self.before_det_t), truncate(self.after_det_t), truncate(det_time), len(info), 0, 0])

      '''' Publish the resultant image after all processes '''
      #self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.detected_frame, "bgr8"))



    except CvBridgeError as e:
      print(e)



  def store_csv(self):
       with open(os.path.join(self.folder, 'ssd_300' + '.csv'), 'w') as fout:
           writer = csv.writer(fout)
           writer.writerows(self.data_file)
       fout.close()
       print('saved')


if __name__ == '__main__':

  def truncate(f):
     return math.floor(f * 10 ** 2) / 10 ** 2

  dio = drone_IO()

  try:
    dio.drone_subscriber()
    dio.store_csv()

  except CvBridgeError as e:
    print(e, 'saif')

  except KeyboardInterrupt:
    print("Shutting down saif")

  except rospy.ROSInterruptException:
    pass

