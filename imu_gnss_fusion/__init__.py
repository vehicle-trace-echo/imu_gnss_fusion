from math import sin, cos, tan, sqrt
from math import radians as rad
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
    LOCAL_GRAV_MAGNTD = 9.8066     
    

    def __init__(self, lat_fix, long_fix, t):
        IMUGNSSFusion.setMeridianCurvRadius(lat_fix, long_fix)
        IMUGNSSFusion.setTransverseCurvRadius(lat_fix, long_fix)
        IMUGNSSFusion.setTimeInterval(t)
        pass

    
    @staticmethod
    def evalJacobianAtState(STATE_NOW:np.ndarray, STATE_NXT:np.ndarray, INPUT_NOW: np.ndarray) ->np.ndarray:
        '''
        '''
        t = IMUGNSSFusion.TIME_INTRVL
        Rn = IMUGNSSFusion.RADIUS_MRIDNL
        Re = IMUGNSSFusion.RADIUS_TRNVRS
        g = IMUGNSSFusion.LOCAL_GRAV_MAGNTD

        psi, vN, vE, vD, h, lat, long, bax, bay, baz,                \
        bgx, bgy, bgz, alfayz, alfazy, alfazx,                      \
        gamayz, gamazy, gamazx, sax, say, saz,                      \
        sgx, sgy, sgz =                                             \
            STATE_NOW[0], STATE_NOW[1], STATE_NOW[2], STATE_NOW[3], \
            STATE_NOW[4], STATE_NOW[5], STATE_NOW[6], STATE_NOW[7], STATE_NOW[8], STATE_NOW[9], \
            STATE_NOW[10], STATE_NOW[11], STATE_NOW[12], STATE_NOW[13], STATE_NOW[14], STATE_NOW[15], \
            STATE_NOW[16], STATE_NOW[17], STATE_NOW[18], STATE_NOW[19], STATE_NOW[20], STATE_NOW[21],\
            STATE_NOW[22], STATE_NOW[23], STATE_NOW[24]
        
        psin, vNn, vEn, vDn, hn, latn, longn, baxn, bayn, bazn,                \
        bgxn, bgyn, bgzn, alfayzn, alfazyn, alfazxn,                      \
        gamayzn, gamazyn, gamazxn, saxn, sayn, sazn,                      \
        sgxn, sgyn, sgzn =                                             \
            STATE_NXT[0], STATE_NXT[1], STATE_NXT[2], STATE_NXT[3], \
            STATE_NXT[4], STATE_NXT[5], STATE_NXT[6], STATE_NXT[7], STATE_NXT[8], STATE_NXT[9], \
            STATE_NXT[10], STATE_NXT[11], STATE_NXT[12], STATE_NXT[13], STATE_NXT[14], STATE_NXT[15], \
            STATE_NXT[16], STATE_NXT[17], STATE_NXT[18], STATE_NXT[19], STATE_NXT[20], STATE_NXT[21],\
            STATE_NXT[22], STATE_NXT[23], STATE_NXT[24]

        ax, ay, az, gx, gy, gz, phi, theta, phin, thetan =  INPUT_NOW[0], INPUT_NOW[1], INPUT_NOW[2], INPUT_NOW[3], \
            INPUT_NOW[4], INPUT_NOW[5], INPUT_NOW[6], INPUT_NOW[7], INPUT_NOW[8], INPUT_NOW[9], \
        
        axc, ayc, azc, gxc, gyc, gzx = None

        JACOBN = np.eye(25)


        # Zeroth Column
        JACOBN[1,0] = t * (-sin(psi)*cos(theta)-sin(psin)*cos(thetan)) * \
                        (axc - g*(sin(theta) + sin(thetan))/2) 

        JACOBN[2,0] = t * (cos(psi)*cos(theta)+cos(psin)*cos(thetan)) * \
                        (axc - g*(sin(theta) + sin(thetan))/2) 

        JACOBN[5,0] = (t / (2 * Rn * hn)) * JACOBN[1,0] 

        JACOBN[6,0] = (t / (2 * Re )) * ( JACOBN[2,0] / (hn * cos(rad(latn)) )   + \
                                    JACOBN[5,0] * vEn * sin(rad(latn)) / (hn * cos(rad(latn)**2) )
                                    )
        

        # First Column
        JACOBN[5,1] = (t / (2 * Rn)) * (1/hn + 1/h)
        JACOBN[6,1] = t * JACOBN[5,1] * vEn * sin(rad(latn)) / (2 * Re * hn * cos(rad(latn))**2)

        # Second Column
        JACOBN[6,2] =  ((t / (2 * Re))) * (1 / (hn * cos(rad(latn))) + 1 / (h * cos(lat)) )

        # Third Column
        JACOBN[4,3] = -t
        JACOBN[5,3] = t**2 * vNn / (2 * Rn * hn**2)
        JACOBN[6,3] = (t / (2 * Re)) *  ( t * vEn / (hn**2 * cos(rad(latn)) ) \
                                     + JACOBN[5,3]*vEn*sin(rad(latn)) / (hn * cos(rad(latn))**2) 
                                     )

        # Fourth Column
        JACOBN[5,4] = (t / (2 * Rn)) * (-vN /h**2 - vNn/hn**2)
        JACOBN[6,4] = (t / (2 * Rn)) * ( JACOBN[5,4]*vEn * sin(rad(latn)) / (hn * cos(latn)**2) -\
                                        vE / (h**2 * cos(rad(lat)) ) - \
                                        vEn / (hn**2 * cos(rad(latn)) )
                                        )

        # Fifth Column
        JACOBN[6,5] = (t / (2 * Rn)) * ( vE * sin(rad(lat)) / (h * cos(lat)**2 ) \
                                        + vEn * sin(rad(latn)) / (hn * cos(latn)**2))
        

        # Seventh Column
        JACOBN[1,7] = - t * sax * (cos(psi)*cos(theta) + cos(psin)*cos(thetan)) / 2
        JACOBN[2,7] = - t * sax * (sin(psi)*cos(theta) + sin(psin)*cos(thetan)) / 2
        JACOBN[3,7] = - t * sax * (-sin(theta) - sin(thetan))/2
        JACOBN[4,7] = JACOBN[3,7] * -t/2
        JACOBN[5,7] = (t / (2 * Rn)) * (JACOBN[1,7]/hn - (JACOBN[4,7]*vNn)/hn**2)
        JACOBN[6,7] = (t / (2 * Re)) * (JACOBN[2,7]/ ( hn * cos(rad(latn)))  - \
                                        (JACOBN[4,7] * vEn / (hn**2 / cos(rad(latn)) )) + \
                                        JACOBN[5,7] * vEn * sin(rad(latn)) / (hn * cos(rad(latn))**2 )
                                        )
                                        
        # Eigth Column
        JACOBN[1,8] = t * say * alfayz * (cos(psi)*cos(theta) + cos(psin)*cos(thetan)) / 2
        JACOBN[2,8] = t * say * alfayz * (sin(psi)*cos(theta) + sin(psin)*cos(thetan)) / 2
        JACOBN[3,8] = t * say * alfayz * (-sin(theta) - sin(thetan))/2
        JACOBN[4,8] = JACOBN[3,8] * -t/2
        JACOBN[5,8] = (t / (2 * Rn)) * (JACOBN[1,8]/hn - (JACOBN[3,8]*vNn)/(2*hn**2))

        JACOBN[6,8] = (t / (2 * Re)) * ((t * JACOBN[3,8] * vEn / (2 * hn**2 / cos(rad(latn)) )) + \
                                        JACOBN[2,8]/ ( hn * cos(rad(latn)))  + \
                                        JACOBN[5,8] * vEn * sin(rad(latn)) / (hn * cos(rad(latn))**2 )
                                        ) 
        
        # Ninth Column
        JACOBN[1,9] = - t * saz * alfazy * (cos(psi)*cos(theta) + cos(psin)*cos(thetan)) / 2
        JACOBN[2,9] = - t * saz * alfazy * (sin(psi)*cos(theta) + sin(psin)*cos(thetan)) / 2
        JACOBN[3,9] = - t * saz * alfazy * (-sin(theta) - sin(thetan))/2
        JACOBN[4,9] = JACOBN[3,9] * -t/2
        JACOBN[5,9] = (t / (2 * Rn)) * (JACOBN[1,9]/hn - (JACOBN[4,9]*vNn)/hn**2)

        JACOBN[6,9] = (t / (2 * Re)) * (JACOBN[2,9]/ ( hn * cos(rad(latn)))  - \
                                        (JACOBN[4,9] * vEn / (hn**2 / cos(rad(latn)) )) + \
                                        JACOBN[5,9] * vEn * sin(rad(latn)) / (hn * cos(rad(latn))**2 )
                                        )



    @staticmethod
    def getMeridianCurvRadius(lat_fix, long_fix):
        '''
        Return R_n (Meridional Raidus of Curvature)
        '''
        return IMUGNSSFusion.RADIUS_EQTRL * (1-IMUGNSSFusion.ECNTRCTY**2) / \
            (1 - IMUGNSSFusion.ECNTRCTY**2 *                    \
            (sin(rad(lat_fix)))**2) ** (3/2)

    @staticmethod
    def getTransverseCurvRadius(lat_fix, long_fix):
        '''
        Return R_e (Transverse Radius of Curvature)
        '''
        return IMUGNSSFusion.RADIUS_EQTRL / sqrt(1-IMUGNSSFusion.ECNTRCTY**2 * \
                (sin(rad(lat_fix)))**2)
    
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