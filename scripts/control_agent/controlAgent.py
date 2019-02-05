#!/usr/bin/env python
from commands_agent.commandsAgent import command


class controller:

    def __init__(self):

        self.errCentroid_x = 0
        self.errCentroid_y = 0
        self.errAlt = 0
        self.errCanvas_x = 0
        self.errCanvas_y = 0
        self.gaz = 0    #altitude
        self.yaw = 0    #yaw angle
        self.theta = 0
        self.phi = 0
        self.com = command() #saif


    def correct_position(self, movement_x,  movement_y, centroid=None, ardrone_altd=0):

        if centroid:
            errCentroid_x = _dst(centroid[0], 320)
            errCentroid_y = _dst(centroid[1], 180)

            if abs(errCentroid_x) > 0.05:
                self.yaw = pid_update(errCentroid_x, self.errCentroid_x, 3) #yaw
                self.errCentroid_x = errCentroid_x
            elif abs(errCentroid_x) <= 0.05:
                self.yaw = 0


            if abs(errCentroid_y) > 0.05:
                self.gaz = pid_update(errCentroid_y, self.errCentroid_y, 2) #alt
                self.errCentroid_y = errCentroid_y
            elif abs(errCentroid_y) <= 0.05:
                self.gaz = 0

            self.theta = movement_x     #Pitch
            self.phi = movement_y       #Roll

            self.com.Velocity(self.theta, self.phi, self.gaz, self.yaw)

def pid_update(error, error_data, v, kp = [0.25, 0.25, 0.0, 0.2], kd = [0.05, 0.05, 0.0, 0.2], ki = [0.25, 0.25, 0.0, 0.2]):
    return kp[v] * error + ki[v] * (error + error_data) + kd[v] * (error - error_data)

def _dst(ctr, siz):
    # if x is - the centroid to the left, else to the right
    return (ctr - siz) / float(siz)



