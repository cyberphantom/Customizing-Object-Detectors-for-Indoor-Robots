#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2, time, os, csv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from cv_bridge import CvBridge, CvBridgeError
from ucf_ardrone_ros.msg import BOX, horizontalXY, filters
from control_agent.controlAgent import controller
from control_agent.commands_agent.commandsAgent import command
from filters_agent.filtersAgent import apply_filters as FL



''' User offline Tracker (re3) '''
#from tracker_agent.re3.tracker import re3_tracker

'''Using Online Trackers'''
#from tracker_agent.onlineTracker import init_tracker

'''Using MOSSE'''
from tracker_agent.mosseTracker import MOSSE


class collector:

  def __init__(self):
    self.bridge = CvBridge()
    self.twist = Twist()

    self.controller = controller()
    self.command = command()
    self.mosse_centroid = None
    #self.online_centroid = None
    #self.re3_centroid = None
    self.move = [0, 0, 0, 0]
    self.h_x = 0
    self.h_y = 0
    self.altd = None
    self.image_pub = rospy.Publisher("/ucf_image_raw", Image)

    self.frame = None
    self.filtered_frame = None
    self.gray_frame = None

    self.pre_filter = None
    self.filters = None

    self.initTime = None
    self.init_bbx = None
    self.mosses = []
    self.re3_bbx = None
    self.drag_rect = None

    self.data_file = []
    self.obj_class = 'test'
    self.prev_time = 0



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

      # Listen to Filters changing
      rospy.Subscriber("/filters", filters, self.filters_calback)



  def filters_calback(self, fl):
      if fl is not None:
          self.filters = fl.fls



  def boxCanvas_callback(self, data):

    if data.data and data.data[0] == -1 and data.data[1] == -1 and data.data[2] == -1 \
            and data.data[3] == -1:

      '''Stop tracking'''
      self.init_bbx = None
      self.twist.linear.x = None
      self.mosses = []
      #self.re3_bbx = None
      self.command.hover()
    elif data.data and data.data[0] != -1 and data.data[1] != -1 and data.data[2] != -1 \
              and data.data[3] != -1:

      '''tracking'''
      if self.frame is not None:

        self.init_bbx = [data.data[0],data.data[1],data.data[2],data.data[3]]

        '''Initialize re3'''
        #tracker.track(self.obj_class, self.frame[:,:,::-1], self.init_bbx)

        '''Uncomment to use online trackers'''
        #self.online_tracker.init(image, (self.init_bbx[0], self.init_bbx[1], self.init_bbx[2],
        # self.init_bbx[3]))

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

      rows, cols = frame.shape[:2]

      folder = '/home/saif/frames/'



      if self.init_bbx is not None:

        if not os.path.exists(os.path.join(folder, self.obj_class)):
            os.makedirs(os.path.join(folder, self.obj_class))

        ticks = time.time()
        delta_time = ticks - self.prev_time

        '''Uncomment to use online trackers'''
        # _, bbx_online = self.online_tracker.update(frame)
        #
        # online_x = int(bbx_online[2]) - ((int(bbx_online[2] - int(bbx_online[0])) /2))
        # online_y = int(bbx_online[3]) - ((int(bbx_online[3] - int(bbx_online[1])) /2))
        #
        # cv2.rectangle(frame,
        #               (int(bbx_online[0]), int(bbx_online[1])),
        #               (int(bbx_online[2]), int(bbx_online[3])),
        #               [0,255,0], 2)
        #
        # self.online_centroid = [online_x, online_y]
        #
        # cv2.line(frame, (online_x - 4, online_y), (online_x + 4, online_y), (200, 50, 50), 2)
        # cv2.line(frame, (online_x, online_y - 4), (online_x, online_y + 4), (200, 50
        #                                                                      , 50), 2)

        '''Uncomment to use MOSSE'''
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for mosse in self.mosses:

            if delta_time >= 0.5:
                ''' Save frames with BBX '''
                file_name = str(ticks).split(".")[0]

                cv2.imwrite(os.path.join(folder, self.obj_class, file_name) + '.png', frame)
                if file_name != str(self.prev_time).split(".")[0]:
                    self.data_file.append(
                        [file_name + '.png',
                         int(mosse.pos[0]) - int(mosse.size[0] / 2),
                         int(mosse.pos[0]) + int(mosse.size[0] / 2),
                         int(mosse.pos[1]) - int(mosse.size[1] / 2),
                         int(mosse.pos[1]) + int(mosse.size[1] / 2),
                         self.obj_class]
)
                self.prev_time = ticks

            mosse.update(gray_frame)
            self.mosse_centroid = mosse.pos
            mosse.draw_state(frame)


        '''re3 tracking call'''
        #self.re3_bbx = tracker.track(self.obj_class, frame[:,:,::-1])

        # if delta_time >= 0.5:
        #
        #     ''' Save objects '''
        #     # cv2.imwrite(os.path.join(folder, self.obj_class, str(ticks).split(".")[0]) +
        #     #             '.png', frame[int(self.re3_bbx[1]):int(self.re3_bbx[3]),
        #     #                    int(self.re3_bbx[0]):int(self.re3_bbx[2])])
        #
        #     ''' Save frames with BBX '''
        #     file_name = str(ticks).split(".")[0]
        #
        #
        #     cv2.imwrite(os.path.join(folder, self.obj_class, file_name) + '.png', frame)
        #     if file_name != str(self.prev_time).split(".")[0]:
        #         self.data_file.append([os.path.join(folder, self.obj_class, file_name) + '.png', cols, rows, self.obj_class, int(self.re3_bbx[0]),
        #                            int(self.re3_bbx[1]), int(self.re3_bbx[2]),
        #                            int(self.re3_bbx[3])])
        #
        #     self.prev_time = ticks
        #
        # cv2.rectangle(frame,
        #               (int(self.re3_bbx[0]), int(self.re3_bbx[1])),
        #               (int(self.re3_bbx[2]), int(self.re3_bbx[3])),
        #               [0,0,255], 2)
        #
        #
        # ''' Create a centroid point just to check accuracy'''
        # re3_x = int(self.re3_bbx[2]) - ((int(self.re3_bbx[2] - int(self.re3_bbx[0])) /2))
        # re3_y = int(self.re3_bbx[3]) - ((int(self.re3_bbx[3] - int(self.re3_bbx[1])) /2))
        # self.re3_centroid = [re3_x, re3_y]
        #
        # cv2.line(frame, (re3_x - 4, re3_y), (re3_x + 4, re3_y), (50, 50, 200), 2)
        # cv2.line(frame, (re3_x, re3_y - 4), (re3_x, re3_y + 4), (50, 50, 200), 2)



      else:
        self.mosse_centroid = None

        with open(os.path.join(folder, self.obj_class + '.csv'), 'w') as fout:
            writer = csv.writer(fout)
            writer.writerows(self.data_file)
        fout.close()

      '''Comment for MOSSE control'''
      self.controller.correct_position(self.h_x, self.h_y, self.mosse_centroid, self.altd)



      '''' Publish the resultant image after all processes '''
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    except CvBridgeError as e:
      print(e)

  '''Uncomment to use MOSSE'''
  def onrect(self, rect): #saif
    if self.frame is not None:
        self.gray_frame = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2GRAY)
    mos = MOSSE(self.gray_frame, rect)
    self.mosses.append(mos)




if __name__ == '__main__':

  #tracker = re3_tracker.Re3Tracker()
  co = collector()
  rospy.init_node('ucf_drone', anonymous=True)
  try:
    co.drone_subscriber()
    rospy.spin()

  except CvBridgeError as e:
    print(e)

  except KeyboardInterrupt:
    print("Shutting down")

  except rospy.ROSInterruptException:
    pass