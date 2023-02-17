import numpy as np
class ComplementaryFilter():
    #Time Constant
    TAU = 0.1
    #Sampling Frequency
    fs = 50
    #Delta T
    dt = 1/fs
    #Alpha
    ALPHA=TAU/(TAU+dt)
    #Gravity
    g=9.81
#Initialize variables
    def __init__(self, rollInit, pitchInit):
        self.roll = rollInit
        self.pitch = pitchInit

    #Compute Gyro Angle for Roll & Pitch
    def gyro_step(self,gx,gy):
        self.roll += gx*ComplementaryFilter.dt
        self.pitch += gy*ComplementaryFilter.dt

    #Compute Accelerometer Angle for Roll & Pitch
    def accel_step(self,ax,ay,az):
        self.theta = np.arcsin(-ax/ComplementaryFilter.g)#Pitch
        self.phi = np.arctan2(-az,ay)#Roll

    #Update Pitch and Roll
    def cf_update(self, ax, ay, az, gx, gy, gz):

        self.gyro_step(gx, gy)
        self.accel_step(ax, ay, az)

        self.rollLast = self.roll
        self.pitchLast = self.pitch

        self.roll = ((1-ComplementaryFilter.ALPHA)*self.phi) + (ComplementaryFilter.ALPHA*self.roll)
        self.pitch = ((1-ComplementaryFilter.ALPHA)*self.theta) + (ComplementaryFilter.ALPHA*self.pitch)