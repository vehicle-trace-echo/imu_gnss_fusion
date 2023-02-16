import math
import numpy as np
from imu_gnss_fusion.extended_kalman_filter import ExtendedKalmanFilter


class IMUGNSSFusion(ExtendedKalmanFilter):
    RADIUS_EQTRL = 6378137.0            # Equitorial Radius of Ellipsoid (WGS84)
    RADIUS_PLR = 6356752.3142           # Polar Radius of Ellipsoid      (WGS84)
    FLTNING = 1 / 298.257223563         # Flattening                     (WGS84)
    ECNTRCTY = 0.0818181908425          # Eccentricity                   (WGS84)
    RADIUS_MRIDNL = False
    RADIUS_TRNVRS = False
    TIME_INTRVL = False        
    
    def __init__(self, lat_fix, long_fix, t):
        IMUGNSSFusion.setMeridianCurvRadius(lat_fix, long_fix)
        IMUGNSSFusion.setTransverseCurvRadius(lat_fix, long_fix)
        IMUGNSSFusion.setTimeInterval(t)
        pass

    @staticmethod
    def evalJacobianAtState(current_state:np.ndarray, next_state:np.ndarray, current_input: np.ndarray) ->np.ndarray:
        '''
        '''
        pass




    @staticmethod
    def getMeridianCurvRadius(lat_fix, long_fix):
        '''
        Return R_n (Meridional Raidus of Curvature)
        '''
        return IMUGNSSFusion.RADIUS_EQTRL * (1-IMUGNSSFusion.ECNTRCTY**2) / \
            (1 - IMUGNSSFusion.ECNTRCTY**2 *                    \
            (math.sin(math.radians(lat_fix)))**2) ** (3/2)

    @staticmethod
    def getTransverseCurvRadius(lat_fix, long_fix):
        '''
        Return R_e (Transverse Radius of Curvature)
        '''
        return IMUGNSSFusion.RADIUS_EQTRL / math.sqrt(1-IMUGNSSFusion.ECNTRCTY**2 * \
                (math.sin(math.radians(lat_fix)))**2)
    
    @classmethod
    def setMeridianCurvRadius(cls, lat_fix, long_fix):
        cls.RADIUS_MRIDNL = cls.getMeridianCurvRadius(lat_fix, long_fix)

    @classmethod
    def setTransverseCurvRadius(cls, lat_fix, long_fix):
        cls.RADIUS_PLR = cls.getTransverseCurvRadius(lat_fix, long_fix)

    @classmethod
    def setTimeInterval(cls, t):
        cls.TIME_INTRVL = t

    @classmethod 
    def clearMeridianCurvRadius(cls):
        cls.RADIUS_MRIDNL = False

    @classmethod 
    def clearTransverseCurvRadius(cls):
        cls.RADIUS_PLR = False

    @classmethod
    def clearTimeInterval(cls):
        cls.TIME_INTRVL = False