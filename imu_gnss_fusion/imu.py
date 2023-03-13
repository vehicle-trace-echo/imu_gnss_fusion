import numpy as np
from config import GRAV_MAGNTD, IMU_CFG


class IMU():
    # IMU data read through 16-bit signed A/D converter
    IMU_MAX = IMU_CFG['range']['imu_max']
    IMU_MIN = IMU_CFG['range']['imu_min']
    IMU_RANGE = IMU_MAX - IMU_MIN

    # Gyro reads in range +- 250 dps
    GYRO_MIN = IMU_CFG['gyro']['range']['gyro_min']
    GYRO_MAX = IMU_CFG['gyro']['range']['gyro_max']
    GYRO_RANGE = GYRO_MAX - GYRO_MIN

    #Pull up from data source
    GRAV_MAGNTD = GRAV_MAGNTD

    # Accelorometer reads in +- 2g
    ACCEL_MIN = IMU_CFG['accel']['range']['accel_min']
    ACCEL_MAX = IMU_CFG['accel']['range']['accel_max']
    ACCEL_RANGE = ACCEL_MAX - ACCEL_MIN

    # Calibration Parameters
    bxa, bya, bza               =   IMU_CFG['accel']['bias']['x'], \
                                    IMU_CFG['accel']['bias']['y'], \
                                    IMU_CFG['accel']['bias']['z']
    
    bxw, byw, bzw               =   IMU_CFG['gyro']['bias']['x'],\
                                    IMU_CFG['gyro']['bias']['y'],\
                                    IMU_CFG['gyro']['bias']['z']
    
    alfayz, alfazy, alfazx      =  IMU_CFG['accel']['misalign']['alpha_yz'],\
                                    IMU_CFG['accel']['misalign']['alpha_zy'],\
                                    IMU_CFG['accel']['misalign']['alpha_zx']
    
    gamayz, gamazy, gamazx      =  IMU_CFG['gyro']['misalign']['gamma_yz'],\
                                    IMU_CFG['gyro']['misalign']['gamma_zy'],\
                                    IMU_CFG['gyro']['misalign']['gamma_zx']
    
    sxa, sya, sza               =   IMU_CFG['accel']['scale']['x'], \
                                    IMU_CFG['accel']['scale']['y'],\
                                    IMU_CFG['accel']['scale']['z'],

    sxw, syw, szw               =   IMU_CFG['gyro']['scale']['x'], \
                                    IMU_CFG['gyro']['scale']['y'],\
                                    IMU_CFG['gyro']['scale']['z']

    # Accel axes misalignment corrections
    Ta = np.array([[1, -alfayz, alfazy],
                        [0, 1, -alfazx],
                        [0, 0, 1]])
    
    # Gyro axes misalignment corrections
    Tw = np.array([[1, -gamayz, gamazy],
                        [0, 1, -gamazx],
                        [0, 0, 1]])

    # Accel scale factor corrections
    Ka = np.array([[sxa, 0, 0],
                        [0, sya, 0],
                        [0, 0, sza]])

    # Gyro scale factor corrections
    Kw = np.array([[sxw, 0, 0],
                        [0, syw, 0],
                        [0, 0, szw]])
    
    # Accel biases 
    ba = np.array([bxa, bya, bza])

    # Gyro biases
    bw = np.array([bxw, byw, bzw])



    @staticmethod
    def normalize_imu(ax, ay, az, gx, gy, gz)->np.ndarray:
        '''
        Given raw readings from ICM20948 16 bit ADC, convert bitfield to
        physical readings (m/s^2 for accel and degrees/s for gyro).
        '''
        accelNorm = IMU.GRAV_MAGNTD * ((np.array([ax, ay, az]) - IMU.IMU_MIN) \
                    / IMU.IMU_RANGE * IMU.ACCEL_RANGE + IMU.ACCEL_MIN)
        gyroNorm =  (np.array([gx, gy, gz]) - IMU.IMU_MIN) / IMU.IMU_RANGE * IMU.GYRO_RANGE + IMU.GYRO_MIN
        return np.concatenate((accelNorm, gyroNorm))
    


    @staticmethod
    def normalize_accel(rawAccelArr:np.ndarray)->np.ndarray:
        '''
        Given raw readings for accelerometer only, convert 16 bit ADC readings to m/s^2
        '''
        return IMU.GRAV_MAGNTD * ((rawAccelArr - IMU.IMU_MIN) \
                    / IMU.IMU_RANGE * IMU.ACCEL_RANGE + IMU.ACCEL_MIN)
    


    @staticmethod
    def normalize_gyro(rawGyroArr:np.ndarray)->np.ndarray:
        '''
        Given raw readings for gyroscope only, convert 16 bit ADC readings to degrees/s
        '''
        return (rawGyroArr - IMU.IMU_MIN) / IMU.IMU_RANGE * IMU.GYRO_RANGE + IMU.GYRO_MIN
    

    @staticmethod
    def calibrate_accel(accelArr: np.ndarray, biasArr=ba)->np.ndarray:
        '''
        Calibrate accelerometer readings to remove deterministic errors
        '''
        return np.matmul(np.matmul(IMU.Ta, IMU.Ka), (accelArr - biasArr))
    

    @staticmethod
    def calibrate_gyro(gyroArr: np.ndarray, biasArr=bw)->np.ndarray:
        '''
        Calibrate gyroscope readings to remove deterministic errors
        '''
        return np.matmul(np.matmul(IMU.Tw, IMU.Kw), (gyroArr - biasArr))