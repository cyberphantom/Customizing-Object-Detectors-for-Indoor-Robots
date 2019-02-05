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

# Initiate Object detection
from object_detection_agent.od_detect_objects import DetectImage

# Initiate Custom Object detection
from object_detection_agent.od_detect_custom_object import DetectCustom

# Use offline tracker




'''Using Onlinre Trackers'''
#from tracker_agent.trackerAgent import init_tracker



class drone_IO:

  '''default object detection'''
  #di = DetectImage()

  '''Detect custom objects'''
  di = DetectCustom()

  def __init__(self):
    self.bridge = CvBridge()
    self.controller = controller()
    self.command = command()
    self.centroid = None
    self.move = [0, 0, 0, 0]
    self.frame = None
    self.detected_image = None
    self.init_bbx = None
    self.h_x = 0
    self.h_y = 0
    self.image_pub = rospy.Publisher("/ucf_image_raw", Image)
    self.dist = rospy.Publisher("/dist_to_obj", Twist)
    self.dXPose = None
    self.dYPose = None
    self.initTime = None
    self.a0, self.d0 = None, None
    self.a1, self.d1 = None, None
    self.a2, self.d2 = None, None
    self.twist = Twist()


    '''Using Onlinre Trackers'''
    #tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    #self.online_tracker = init_tracker(4)


    '''Using Object Detection'''




  # Define Subscribers
  def drone_subscriber(self):
      rospy.Subscriber("/ardrone/image_raw",Image, self.callback)
      rospy.Subscriber("/ardrone/navdata", Navdata, self.ReceiveNavdata)

      #Initialization BBox
      rospy.Subscriber("/box", BOX, self.boxCanvas)
      rospy.Subscriber("/ucf_drone_pose", Twist, self.pose_callback)
      rospy.Subscriber("/horizontalxy", horizontalXY, self.auto_movement_callback)


  def pose_callback(self, data):
    self.dXPose = data.linear.x
    self.dYPose = data.linear.y
    self.dYaw = data.angular.z



  def boxCanvas(self, data):


    if data.data and data.data[0] == -1 and data.data[1] == -1 and data.data[2] == -1 \
            and data.data[3] == -1:
      #Stop tracking
      self.init_bbx = None
      self.a0, self.a1, self.a2 = None, None, None
      self.twist.linear.x = None
      self.command.hover()



    elif data.data and data.data[0] != -1 and data.data[1] != -1 and data.data[2] != -1 \
              and data.data[3] != -1:
      #tracking
      if self.frame is not None:
        image = self.frame.copy()
        self.init_bbx = [data.data[0],data.data[1],data.data[2],data.data[3]]
        tracker.track('ball', image[:,:,::-1], self.init_bbx)

        '''Uncomment to use online trackers'''
        #self.online_tracker.init(image, (self.init_bbx[0],self.init_bbx[1],self.init_bbx[2],self.init_bbx[3]))

        self.initTime = time.time()

  def auto_movement_callback(self, hxy):
    self.h_x = hxy.x
    self.h_y = hxy.y


  def ReceiveNavdata(self, navdata):
      self.altd = navdata.altd


  def filter_state_callback(self, state):
      self.position = [state.x, state.y,state.z, state.yaw]


  def callback(self, data):
    try:

      self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

      #Processing

      self.detected_image = self.frame.copy()

      '''Call detection before tracking if requested'''
      tracked_image = self.di.run_detect(self.detected_image)

      if self.init_bbx is not None:

        bbx = tracker.track('ball', tracked_image[:,:,::-1])
        cv2.rectangle(tracked_image,
                      (int(bbx[0]), int(bbx[1])),
                      (int(bbx[2]), int(bbx[3])),
                      [0,0,255], 2)

        '''Uncomment to use online trackers'''
        # _, bbx_online = self.online_tracker.update(tracked_image)
        # cv2.rectangle(tracked_image,
        #               (int(bbx_online[0]), int(bbx_online[1])),
        #               (int(bbx_online[2]), int(bbx_online[3])),
        #               [0,255,0], 2)

        deltaW = int(bbx[2]) - int(bbx[0])
        deltaH = int(bbx[3]) - int(bbx[1])
        area = deltaW * deltaH

        if self.a0 is None and time.time() - self.initTime > 4 and self.dYaw < 0.3:
          self.a0 = deltaW * deltaH
          self.d0 = self.dXPose

        self.centroid = int(deltaW / 2) + int(bbx[0]), int(deltaH / 2) + int(bbx[1])
        cv2.rectangle(tracked_image, (self.centroid[0] - 1, self.centroid[1] - 1), (self.centroid[0] + 1,
                                                                                    self.centroid[1] + 1), [0,0,255], -1)
        self.controller.correct_position(self.h_x, self.h_y, self.centroid, self.altd)

        if self.a0 is not None:
          if self.a1 is None and area > (self.a0 + (1 * self.a0)) and abs(self.dYaw) < 0.3 \
                  and abs(self.dYPose) < 0.3:
            self.a1 = area
            self.d1 = self.dXPose


          if self.a2 is None and area > (self.a0 + (3 * self.a0)) and abs(self.dYaw) < 0.3 \
                  and abs(self.dYPose) < 0.3:
            self.a2 = area
            self.d2 = self.dXPose
            Dist = findDistance(self.a0, self.d0, self.a1, self.d1, self.a2, self.d2)


            self.twist.linear.x = Dist
            self.twist.linear.y = 0
            self.twist.linear.z = 0
            self.twist.angular.x = 0
            self.twist.angular.y = 0
            self.twist.angular.z = 0
            self.dist.publish(self.twist)

      self.image_pub.publish(self.bridge.cv2_to_imgmsg(tracked_image, "bgr8"))
    except CvBridgeError as e:
      print(e)




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

