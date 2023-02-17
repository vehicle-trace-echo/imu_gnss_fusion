import numpy as np

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
bxa, bya, bza               =   0.0335241, 0.0335241, 0.08829979
bxw, byw, bzw               =   -0.6584513260335345, 1.202140150793541, -0.1227491118389093
alfayz, alfazy, alfazx      =   0.0335241,0.00295066,0.00359227,
gamayz, gamazy, gamazx      =   0, 0, 0
sxa, sya, sza               =   1.00168651, 0.99445974, 0.98556708,
sxw, syw, szw               =   1, 1, 1



Ta = np.array([1, -alfayz, alfazy],
                    [0, 1, -alfazx],
                    [0, 0, 1])
Tw = np.array([[1, -gamayz, gamazy],
                    [0, 1, -gamazx],
                    [0, 0, 1]])

Ka = np.array([[sxa, 0, 0],
                    [0, sya, 0],
                    [0, 0, sza]])

Kw = np.array([[sxw, 0, 0],
                    [0, syw, 0],
                    [0, 0, szw]])

ba = np.array([bxa, bya, bza])

bw = np.array([bxw, byw, bzw])



def normalize_imu(ax, ay, az, gx, gy, gz)->np.ndarray:
    accelNorm = GRAV_MAGNTD * ((np.array([ax, ay, az]) - IMU_MIN) \
                 / IMU_RANGE * ACCEL_RANGE + ACCEL_MIN)
    gyroNorm =  (np.array([gx, gy, gz]) - IMU_MIN) / IMU_RANGE * GYRO_RANGE + GYRO_MIN
    return np.concatenate((accelNorm, gyroNorm))



def calibrate_accel(accelArr: np.ndarray)->np.ndarray:
    return np.matmul(np.matmul(Ta, Ka), (accelArr - ba))


def calibrate_gyro(gyroArr: np.ndarray)->np.ndarray:
    return np.matmul(np.matmul(Tw, Kw), (gyroArr - bw))
