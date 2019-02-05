#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2, time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from cv_bridge import CvBridge, CvBridgeError
from ucf_ardrone_ros.msg import BOX, horizontalXY, filters
from control_agent.controlAgent import controller
from control_agent.commands_agent.commandsAgent import command
from filters_agent.filtersAgent import apply_filters as FL


''' Using Object Detection API '''
#from object_detection_agent.research.od_detect_objects import DetectImage

''' Using Object Detection API with custom objects '''
from object_detection_agent.research.od_detect_custom_object import DetectCustom

''' Using Object Detection API with mix objects '''
#from object_detection_agent.research.od_two_models import DetectMix


'''Using MOSSE'''
from tracker_agent.mosseTracker import MOSSE


class drone_IO:

  '''default object detection'''
  #di = DetectImage()

  '''Detect custom objects'''
  di = DetectCustom()

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
    self.image_pub = rospy.Publisher("/ucf_image_raw", Image)
    self.dist = rospy.Publisher("/dist_to_obj", Twist)
    self.initTime = None
    self.a0, self.d0 = None, None
    self.a1, self.d1 = None, None
    self.a2, self.d2 = None, None
    self.altd = None
    self.twist = Twist()

    self.mosses = []
    self.mosse_centroid = None
    self.drag_rect = None
    self.prev_time = 0
    self.detect_time = 0
    self.tracking_time = 0
    #self.online_centroid = None
    #self.re3_centroid = None
    #self.re3_bbx = None

    '''Using MOSSE'''
    from tracker_agent.mosseTracker import MOSSE



  '''Define Subscribers'''
  def drone_subscriber(self):

      # Subscribe to ardrone_autonomy
      rospy.Subscriber("/ardrone/image_raw",Image, self.callback)
      rospy.Subscriber("/ardrone/navdata", Navdata, self.receiveNavdata_callback)

      rospy.Subscriber("/horizontalxy", horizontalXY, self.auto_movement_callback)

      # Listen to Filters changing
      rospy.Subscriber("/filters", filters, self.filters_calback)



  def filters_calback(self, fl):
      if fl is not None:
          self.filters = fl.fls


  def receiveNavdata_callback(self, navdata):
      self.altd = navdata.altd


  '''Create function that get frame and apply detection'''


  def callback(self, data):
    try:

      self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

      '''Processing'''

      if self.filters is not None:
        self.filtered_frame = FL(self.frame, self.filters)
        frame = self.filtered_frame
      else:
        frame = self.frame.copy()

      '''Call detection before tracking if requested'''
      detected = False
      ticks = time.time()
      detect_delta_time = ticks - self.detect_time
      boxes = None

      '''Change to use interval pre preprocess'''
      if detect_delta_time >= 0.249:
          self.detected_frame, boxes = self.di.run_detect(frame)

          if boxes is not None:
              detected = True
              self.tracking_time = time.time()

      tracking_delta_time = time.time() - self.tracking_time
      if tracking_delta_time <= 0.25 and result_from_iter == None:
          '''run tracker'''
          if boxes is not None and frame is not None:
              # Check to run twice or create 2D matrix
              for i in boxes.length:
                self.onrect(tuple(self.bounding_box(boxes[i])))

              gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              for mosse in self.mosses:
                mosse.update(gray_frame)
                self.mosse_centroid = mosse.pos
                mosse.draw_state(frame)



      if detect_delta_time > 0.25:
          '''Deactivate tracker'''
          self.mosses = []
          detected_objects = []

      self.detect_time = ticks


      '''' Publish the resultant image after all processes '''

      self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.detected_frame, "bgr8"))

    except CvBridgeError as e:
      print(e)



  '''Uncomment to use MOSSE'''
  def onrect(self, rect): #saif
    if self.frame is not None:
        self.gray_frame = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2GRAY)
    mos = MOSSE(self.gray_frame, rect)
    self.mosses.append(mos)

  def bounding_box(self, box):
      minx = int(box[0] * 640)
      miny = int(box[1] * 360)
      maxx = int(box[2] * 640)
      maxy = int(box[3] * 360)
      #deltaW = maxx - minx
      #deltaH = maxy - miny
      #box_area = deltaW * deltaH
      #box_ratio = float(box_area) / float(640 * 360)
      #centroid = [int(deltaW / 2) + minx, int(deltaH / 2) + miny]
      return minx, miny, maxx, maxy



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

