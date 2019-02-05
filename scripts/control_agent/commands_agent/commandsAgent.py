#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Empty

class command:

    def __init__(self):
        self._reset = rospy.Publisher('/ardrone/reset', Empty, queue_size=1)
        self._velocity = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.isTakeoff = False



    def reset(self):
        self._reset.publish(Empty())

    def hover(self):
        self._velocity.publish(Twist(Vector3(0,0,0),Vector3(0,0,0)))

    def Velocity(self, theta=0, phi=0, gaz=0, yaw=0):
        cmd = Twist()
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = -yaw
        cmd.linear.x = theta
        cmd.linear.y = phi
        cmd.linear.z = -gaz

        self._velocity.publish(cmd)
