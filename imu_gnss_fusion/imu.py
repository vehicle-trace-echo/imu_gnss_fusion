import numpy as np


class IMU():
    # IMU data read through 16-bit signed A/D converter
    IMU_MAX = 32767
    IMU_MIN = -32768
    IMU_RANGE = IMU_MAX - IMU_MIN

    # Gyro reads in range +- 1000 dps
    GYRO_MIN = -250
    GYRO_MAX = 250
    GYRO_RANGE = GYRO_MAX - GYRO_MIN

    #Pull up from data source
    GRAV_MAGNTD = 9.8066

    # Accelorometer reads in +- 2g
    ACCEL_MIN = -2
    ACCEL_MAX = 2
    ACCEL_RANGE = ACCEL_MAX - ACCEL_MIN

    # Calibration Parameters
    alfayz, alfazy, alfazx      =   0.0335241,0.00295066,0.00359227,
    gamayz, gamazy, gamazx      =   0, 0, 0
    sxa, sya, sza               =   1.00168651, 0.99445974, 0.98556708,
    sxw, syw, szw               =   1, 1, 1



    Ta = np.array([[1, -alfayz, alfazy],
                        [0, 1, -alfazx],
                        [0, 0, 1]])
    Tw = np.array([[1, -gamayz, gamazy],
                        [0, 1, -gamazx],
                        [0, 0, 1]])

    Ka = np.array([[sxa, 0, 0],
                        [0, sya, 0],
                        [0, 0, sza]])

    Kw = np.array([[sxw, 0, 0],
                        [0, syw, 0],
                        [0, 0, szw]])


    @staticmethod
    def normalize_imu(ax, ay, az, gx, gy, gz)->np.ndarray:
        accelNorm = IMU.GRAV_MAGNTD * ((np.array([ax, ay, az]) - IMU.IMU_MIN) \
                    / IMU.IMU_RANGE * IMU.ACCEL_RANGE + IMU.ACCEL_MIN)
        gyroNorm =  (np.array([gx, gy, gz]) - IMU.IMU_MIN) / IMU.IMU_RANGE * IMU.GYRO_RANGE + IMU.GYRO_MIN
        return np.concatenate((accelNorm, gyroNorm))
    
    @staticmethod
    def normalize_accel(rawAccelArr:np.ndarray)->np.ndarray:
        return IMU.GRAV_MAGNTD * ((rawAccelArr - IMU.IMU_MIN) \
                    / IMU.IMU_RANGE * IMU.ACCEL_RANGE + IMU.ACCEL_MIN)
    

    @staticmethod
    def normalize_gyro(rawGyroArr:np.ndarray)->np.ndarray:
        return (rawGyroArr - IMU.IMU_MIN) / IMU.IMU_RANGE * IMU.GYRO_RANGE + IMU.GYRO_MIN
    

    @staticmethod
    def calibrate_accel(accelArr: np.ndarray, biasArr:np.ndarray)->np.ndarray:
        return np.matmul(np.matmul(IMU.Ta, IMU.Ka), (accelArr - biasArr))
    
    @staticmethod
    def calibrate_gyro(gyroArr: np.ndarray, biasArr: np.ndarray)->np.ndarray:
        return np.matmul(np.matmul(IMU.Tw, IMU.Kw), (gyroArr - biasArr))