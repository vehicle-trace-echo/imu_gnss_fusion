import numpy as np
class ComplementaryFilter():
    #Time Constant
    TAU = 0.1
    #Sampling Frequency
    fs = 100
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
    def gyro_step(self,gyroArr):
        self.roll += gyroArr[0]*ComplementaryFilter.dt
        self.pitch += gyroArr[1]*ComplementaryFilter.dt

    #Compute Accelerometer Angle for Roll & Pitch
    def accel_step(self, accelArr):
        self.theta = np.arcsin(accelArr[0]/ComplementaryFilter.g)#Pitch
        self.phi = np.arctan2(-1*accelArr[1], -1*accelArr[2])#Roll

    #Update Pitch and Roll
    def cf_update(self, accelArr, gyroArr):

        self.gyro_step(gyroArr)
        self.accel_step(accelArr)

        self.rollLast = self.roll
        self.pitchLast = self.pitch

        self.roll = ((1-ComplementaryFilter.ALPHA)*self.phi) + (ComplementaryFilter.ALPHA*self.roll)
        self.pitch = ((1-ComplementaryFilter.ALPHA)*self.theta) + (ComplementaryFilter.ALPHA*self.pitch)

    def cf_update_INS(self, accArr, pitch_INS,  roll_INS):

        self.accel_step(accArr)
        self.roll = ((1-ComplementaryFilter.ALPHA)*self.phi) + (ComplementaryFilter.ALPHA*roll_INS)
        self.pitch = ((1-ComplementaryFilter.ALPHA)*self.theta) + (ComplementaryFilter.ALPHA*pitch_INS)
        return self.pitch, self.roll