from math import sin, cos, tan, sqrt, asin, atan2, degrees, pi
from math import radians as rad
from ins import INS
import numpy as np



class ImuGnssEsEKF():

    def __init__(self,
            init_atitude: np.ndarray,
            init_velocity: np.ndarray,
            init_position: np.ndarray,
            init_bias_accel: np.ndarray,
            init_bias_gyro:np.ndarray,
            sample_rate: int):
        self.ins = INS(init_atitude,
                       init_velocity,
                       init_position,
                       init_bias_accel,
                       init_bias_gyro,
                       sample_rate)
        pass

    def EsEKF_StateTransition(self,
                              accel_vectr: np.ndarray,
                              gyro_vectr:np.ndarray):
        pass