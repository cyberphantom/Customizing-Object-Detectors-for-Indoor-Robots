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
from object_detection_agent.research.od_detect_objects import DetectImage

''' Using Object Detection API with custom objects '''
#from object_detection_agent.od_detect_custom_object import DetectCustom

''' User offline Tracker (re3) '''
from tracker_agent.re3.tracker import re3_tracker

''' Using measurement agent '''
from measurements_agent.measurementsAgent import findDistance


'''Using Onlinre Trackers'''
#from tracker_agent.onlineTracker import init_tracker


'''Using MOSSE'''
from tracker_agent.mosseTracker import MOSSE


class drone_IO:

  '''default object detection'''
  di = DetectImage()

  '''Detect custom objects'''
  #di = DetectCustom()

  def __init__(self):
    self.bridge = CvBridge()
    self.controller = controller()
    self.command = command()
    self.centroid = None
    self.move = [0, 0, 0, 0]
    self.frame = None
    self.detected_frame = None
    self.filtered_frame = None
    self.gray_frame = None
    self.pre_filter = None
    self.filters = None
    self.init_bbx = None
    self.h_x = 0
    self.h_y = 0
    self.image_pub = rospy.Publisher("/ucf_image_raw", Image)
    self.dist = rospy.Publisher("/dist_to_obj", Twist)
    self.dXPose = None
    self.dYPose = None
    self.dYaw = None
    self.initTime = None
    self.a0, self.d0 = None, None
    self.a1, self.d1 = None, None
    self.a2, self.d2 = None, None
    self.mosses = []
    self.re3_bbx = None
    self.drag_rect = None
    self.twist = Twist()


    '''Using Onlinre Trackers'''
    #self.online_tracker = init_tracker(4)



  '''Define Subscribers'''
  def drone_subscriber(self):

      # Subscribe to ardrone_autonomy
      rospy.Subscriber("/ardrone/image_raw",Image, self.callback)
      rospy.Subscriber("/ardrone/navdata", Navdata, self.receiveNavdata_callback)

      # If a target asigned, we initialize a BBox and coresponding Subscriber
      rospy.Subscriber("/box", BOX, self.boxCanvas_callback)
      rospy.Subscriber("/horizontalxy", horizontalXY, self.auto_movement_callback)

      # Listen to ORB SLAM ardrone pose
      rospy.Subscriber("/ucf_drone_pose", Twist, self.pose_callback)

      # Listen to Filters changing
      rospy.Subscriber("/filters", filters, self.filters_calback)



  def pose_callback(self, data):
    self.dXPose = data.linear.x
    self.dYPose = data.linear.y
    self.dYaw = data.angular.z



  def filters_calback(self, fl):
      if fl is not None:
          self.filters = fl.fls



  def boxCanvas_callback(self, data):

    if data.data and data.data[0] == -1 and data.data[1] == -1 and data.data[2] == -1 \
            and data.data[3] == -1:
      '''Stop tracking'''
      self.init_bbx = None
      self.a0, self.a1, self.a2 = None, None, None
      self.twist.linear.x = None
      self.mosses = []
      self.re3_bbx = None
      self.command.hover()

    elif data.data and data.data[0] != -1 and data.data[1] != -1 and data.data[2] != -1 \
              and data.data[3] != -1:
      '''tracking'''
      if self.frame is not None:
        image = self.frame.copy()
        self.init_bbx = [data.data[0],data.data[1],data.data[2],data.data[3]]

        '''Initialize re3'''

        tracker.track('ball', image[:,:,::-1], self.init_bbx)


        '''Uncomment to use online trackers'''
        #self.online_tracker.init(image, (self.init_bbx[0],self.init_bbx[1],self.init_bbx[2],self.init_bbx[3]))

        '''Uncomment to use MOSSE'''
        self.onrect(tuple(data.data))

        self.initTime = time.time()



  def auto_movement_callback(self, hxy):
    self.h_x = hxy.x
    self.h_y = hxy.y



  def receiveNavdata_callback(self, navdata):
      self.altd = navdata.altd



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
      self.detected_frame = self.di.run_detect(frame)
      #self.detected_frame = frame


      if self.init_bbx is not None:

        '''Uncomment to use online trackers'''
        # _, bbx_online = self.online_tracker.update(tracked_image)
        # cv2.rectangle(tracked_image,
        #               (int(bbx_online[0]), int(bbx_online[1])),
        #               (int(bbx_online[2]), int(bbx_online[3])),
        #               [0,255,0], 2)

        '''Uncomment to use MOSSE'''
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for mosse in self.mosses:
            mosse.update(gray_frame)
            self.centroid = mosse.pos
            mosse.draw_state(self.detected_frame)

        else:
            self.centroid = None




        '''re3 tracking call'''
        self.re3_bbx = tracker.track('ball', frame[:,:,::-1])
        cv2.rectangle(self.detected_frame,
                      (int(self.re3_bbx[0]), int(self.re3_bbx[1])),
                      (int(self.re3_bbx[2]), int(self.re3_bbx[3])),
                      [0,0,255], 2)



        deltaW = int(self.re3_bbx[2]) - int(self.re3_bbx[0])
        deltaH = int(self.re3_bbx[3]) - int(self.re3_bbx[1])
        area = deltaW * deltaH

        ''' Create a centroid point just to check accuracy'''
        # self.centroid = int(deltaW / 2) + int(bbx[0]), int(deltaH / 2) + int(bbx[1])

        # cv2.rectangle(tracked_frame, (self.centroid[0] - 1, self.centroid[1] - 1), (self.centroid[0] + 1,
        #                                                                             self.centroid[1] + 1), [0,0,255], -1)



        self.controller.correct_position(self.h_x, self.h_y, self.centroid, self.altd)


        '''distance and size measurements - Assign first sizing point '''
        if self.a0 is None and time.time() - self.initTime > 4 and self.dYaw < 0.3:
          self.a0 = deltaW * deltaH
          self.d0 = self.dXPose

        ''' Assign second sizing point '''
        if self.a0 is not None:
          if self.a1 is None and area > (self.a0 + (1 * self.a0)) and abs(self.dYaw or 9) < 0.3 \
                  and abs(self.dYPose) < 0.3:
            self.a1 = area
            self.d1 = self.dXPose


          if self.a2 is None and area > (self.a0 + (3 * self.a0)) and abs(self.dYaw or 9) < 0.3 \
                  and abs(self.dYPose) < 0.3:
            self.a2 = area
            self.d2 = self.dXPose

            ''' Find the distance to target '''
            Dist = findDistance(self.a0, self.d0, self.a1, self.d1, self.a2, self.d2)

            ''' Publish the distance '''
            self.twist.linear.x = Dist
            self.twist.linear.y = 0
            self.twist.linear.z = 0
            self.twist.angular.x = 0
            self.twist.angular.y = 0
            self.twist.angular.z = 0
            self.dist.publish(self.twist)


      '''' Publish the resultant image after all processes '''
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.detected_frame, "bgr8"))

    except CvBridgeError as e:
      print(e)


  def onrect(self, rect): #saif
    if self.frame is not None:
        self.gray_frame = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2GRAY)
    mosse = MOSSE(self.gray_frame, rect)
    self.mosses.append(mosse)


if __name__ == '__main__':
  tracker = re3_tracker.Re3Tracker()
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

