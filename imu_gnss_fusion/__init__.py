from math import sin, cos, tan, sqrt, asin, atan2
from math import rad as rad
import numpy as np
from imu_gnss_fusion.extended_kalman_filter import ExtendedKalmanFilter


class IMUGNSSFusion(ExtendedKalmanFilter):
    RADIUS_EQTRL = 6378137.0            # Equitorial Radius of Ellipsoid (WGS84)
    RADIUS_PLR = 6356752.3142           # Polar Radius of Ellipsoid      (WGS84)
    FLTNING = 1 / 298.257223563         # Flattening                     (WGS84)
    ECNTRCTY = 0.0818181908425          # Eccentricity                   (WGS84)
    INS.RADIUS_MRIDNL = False
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
        INS.RADIUS_MRIDNL = IMUGNSSFusion.INS.RADIUS_MRIDNL
        INS.RADIUS_TRNVRS = IMUGNSSFusion.RADIUS_TRNVRS
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

        JACOBN[5,0] = (t / (2 * INS.RADIUS_MRIDNL * hn)) * JACOBN[1,0] 

        JACOBN[6,0] = (t / (2 * INS.RADIUS_TRNVRS )) * ( JACOBN[2,0] / (hn * cos(rad(latn)) )   + \
                                    JACOBN[5,0] * vEn * sin(rad(latn)) / (hn * cos(rad(latn)**2) )
                                    )
        

        # First Column
        JACOBN[5,1] = (t / (2 * INS.RADIUS_MRIDNL)) * (1/hn + 1/h)
        JACOBN[6,1] = t * JACOBN[5,1] * vEn * sin(rad(latn)) / (2 * INS.RADIUS_TRNVRS * hn * cos(rad(latn))**2)

        # Second Column
        JACOBN[6,2] =  ((t / (2 * INS.RADIUS_TRNVRS))) * (1 / (hn * cos(rad(latn))) + 1 / (h * cos(lat)) )

        # Third Column
        JACOBN[4,3] = -t
        JACOBN[5,3] = t**2 * vNn / (2 * INS.RADIUS_MRIDNL * hn**2)
        JACOBN[6,3] = (t / (2 * INS.RADIUS_TRNVRS)) *  ( t * vEn / (hn**2 * cos(rad(latn)) ) \
                                     + JACOBN[5,3]*vEn*sin(rad(latn)) / (hn * cos(rad(latn))**2) 
                                     )

        # Fourth Column
        JACOBN[5,4] = (t / (2 * INS.RADIUS_MRIDNL)) * (-vN /h**2 - vNn/hn**2)
        JACOBN[6,4] = (t / (2 * INS.RADIUS_MRIDNL)) * ( JACOBN[5,4]*vEn * sin(rad(latn)) / (hn * cos(latn)**2) -\
                                        vE / (h**2 * cos(rad(lat)) ) - \
                                        vEn / (hn**2 * cos(rad(latn)) )
                                        )

        # Fifth Column
        JACOBN[6,5] = (t / (2 * INS.RADIUS_MRIDNL)) * ( vE * sin(rad(lat)) / (h * cos(lat)**2 ) \
                                        + vEn * sin(rad(latn)) / (hn * cos(latn)**2))
        

        # Seventh Column
        JACOBN[1,7] = - t * sax * (cos(psi)*cos(theta) + cos(psin)*cos(thetan)) / 2
        JACOBN[2,7] = - t * sax * (sin(psi)*cos(theta) + sin(psin)*cos(thetan)) / 2
        JACOBN[3,7] = - t * sax * (-sin(theta) - sin(thetan))/2
        JACOBN[4,7] = JACOBN[3,7] * -t/2
        JACOBN[5,7] = (t / (2 * INS.RADIUS_MRIDNL)) * (JACOBN[1,7]/hn - (JACOBN[4,7]*vNn)/hn**2)
        JACOBN[6,7] = (t / (2 * INS.RADIUS_TRNVRS)) * (JACOBN[2,7]/ ( hn * cos(rad(latn)))  - \
                                        (JACOBN[4,7] * vEn / (hn**2 / cos(rad(latn)) )) + \
                                        JACOBN[5,7] * vEn * sin(rad(latn)) / (hn * cos(rad(latn))**2 )
                                        )
                                        
        # Eigth Column
        JACOBN[1,8] = t * say * alfayz * (cos(psi)*cos(theta) + cos(psin)*cos(thetan)) / 2
        JACOBN[2,8] = t * say * alfayz * (sin(psi)*cos(theta) + sin(psin)*cos(thetan)) / 2
        JACOBN[3,8] = t * say * alfayz * (-sin(theta) - sin(thetan))/2
        JACOBN[4,8] = JACOBN[3,8] * -t/2
        JACOBN[5,8] = (t / (2 * INS.RADIUS_MRIDNL)) * (JACOBN[1,8]/hn - (JACOBN[3,8]*vNn)/(2*hn**2))

        JACOBN[6,8] = (t / (2 * INS.RADIUS_TRNVRS)) * ((t * JACOBN[3,8] * vEn / (2 * hn**2 / cos(rad(latn)) )) + \
                                        JACOBN[2,8]/ ( hn * cos(rad(latn)))  + \
                                        JACOBN[5,8] * vEn * sin(rad(latn)) / (hn * cos(rad(latn))**2 )
                                        ) 
        
        # Ninth Column
        JACOBN[1,9] = - t * saz * alfazy * (cos(psi)*cos(theta) + cos(psin)*cos(thetan)) / 2
        JACOBN[2,9] = - t * saz * alfazy * (sin(psi)*cos(theta) + sin(psin)*cos(thetan)) / 2
        JACOBN[3,9] = - t * saz * alfazy * (-sin(theta) - sin(thetan))/2
        JACOBN[4,9] = JACOBN[3,9] * -t/2
        JACOBN[5,9] = (t / (2 * INS.RADIUS_MRIDNL)) * (JACOBN[1,9]/hn - (JACOBN[4,9]*vNn)/hn**2)

        JACOBN[6,9] = (t / (2 * INS.RADIUS_TRNVRS)) * (JACOBN[2,9]/ ( hn * cos(rad(latn)))  - \
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
        cls.INS.RADIUS_MRIDNL = cls.getMeridianCurvRadius(lat_fix, long_fix)

    @classmethod
    def setTransverseCurvRadius(cls, lat_fix, long_fix):
        cls.RADIUS_PLR = cls.getTransverseCurvRadius(lat_fix, long_fix)

    @classmethod
    def setTimeInterval(cls, t):
        cls.TIME_INTRVL = t

    @classmethod 
    def clearMeridianCurvRadius(cls):
        cls.INS.RADIUS_MRIDNL = False

    @classmethod 
    def clearTransverseCurvRadius(cls):
        cls.RADIUS_PLR = False

    @classmethod
    def clearTimeInterval(cls):
        cls.TIME_INTRVL = False






class INS():
    RADIUS_EQTRL = 6378137.0            # Equitorial Radius of Ellipsoid (WGS84)
    RADIUS_PLR = 6356752.3142           # Polar Radius of Ellipsoid      (WGS84)
    FLTNING = 1 / 298.257223563         # Flattening                     (WGS84)
    ECNTRCTY = 0.0818181908425          # Eccentricity                   (WGS84)
    EATH_ROTN_RATE = 7.292115E-5        # Angular rotation rate of earth in rad/s
    RADIUS_MRIDNL = False               # RN
    RADIUS_TRNVRS = False               # RE  
    TIME_INTRVL = False   
    LOCAL_GRAV_MAGNTD = 9.8066     
    
    '''
    Inertial Navigation System (INS) mechanizes the IMU in the Local Navigation Frame.

    INS works in parallel with and ES-EKF, for Atitude, Veloctiy, Position corrections, and
    IMU biases.

    The Mechanization is based on formulas found in 'Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems' - Paul D Groves.
    '''
    def __init__(self, init_atitude: np.ndarray,
                    init_velocity: np.ndarray,
                    init_position: np.ndarray,
                    init_bias_accel: np.ndarray,
                    init_bias_gyro:np.ndarray,
                    sample_rate: int):
        INS.setMeridianCurvRadius(init_position[1], init_position[2])
        INS.setTransverseCurvRadius(init_position[1], init_position[2])
        INS.setTimeInterval(1/sample_rate)

        self.init_states(init_atitude,
                    init_velocity,
                    init_position,
                    init_bias_accel,
                    init_bias_gyro)
        self.init_nexts()


    def init_states(self, init_atitude: np.ndarray,
                    init_velocity: np.ndarray,
                    init_position: np.ndarray,
                    init_bias_accel: np.ndarray,
                    init_bias_gyro:np.ndarray):
        self.atitude = init_atitude
        self.coord_trnsfrm = INS.get_coord_transfrm_frm_atitude(init_atitude)
        self.veloctiy = init_velocity
        self.position = init_position
        self.bias_accel = init_bias_accel
        self.bias_gyro = init_bias_gyro
    

    def init_nexts(self):
        self.next_atitude = np.zeros(3)
        self.coord_trnsfrm = np.zeros((3,3))
        self.next_velocity = np.zeros(3)
        self.next_position = np.zeros(3)
        self.next_bias_accel = np.zeros(3)
        self.next_bais_gyro = np.zeros(3)


    def init_crrctns(self):
        self.atitude_crrctn = np.zeros(3)
        self.velocity_crrctn = np.zeros(3)
        self.position_crrctn = np.zeros(3)


    def update_atitude(
                self,
                acc_vectr:np.ndarray,
                gyro_vectr:np.ndarray,
                comp_filter=None, 
                )->np.ndarray:
        
        self.lat_rad = rad(self.position[1])
        
        self.next_coord_trnsfrm = self.updt_coord_trnsfrm(gyro_vectr)                               # Get next coodinate transform matrix
        self.next_coord_trnsfrm = np.matmul(
                                (np.eye(3) - INS.get_skew_sym_mat(self.atitude_corectn_vectr)),     # Apply atitude correction (small angle approx)
                                    self.coord_trnsfrm_mat)
        self.next_atitude = INS.get_attitude_frm_coord_trnsfrm(self.next_coord_trnsfrm)             # Get corrected atitude       
    

    def updt_coord_trnsfrm(self,
            gyro_vectr:np.ndarray):
    
        atitude_updt_mat = INS.get_atitude_updt_mat(gyro_vectr)
        earth_rot_mat = INS.get_earth_rot_mat(self.lat_rat)
        trnsprt_mat = INS.get_trnsprt_mat(self.position[0], self.lat_rad,
                                          self.velocity[0], self.velocity[1])
        return np.matmul(self.coord_trnsfrm_mat, atitude_updt_mat) - \
                (np.matmul((earth_rot_mat+trnsprt_mat), self.coord_trnsfrm_mat) * INS.TIME_INTRVL)


    @staticmethod
    def get_atitude_updt_mat(gyro_vectr:np.ndarray)->np.ndarray:
        '''
        Fourth order approximation solver for Euler Rates (transformation matrix) from Gyro rates
        '''
        atitude_incrmnt_norm =  np.linalg.norm(INS.get_atitude_incrmnt_vectr(gyro_vectr))
        atitude_incrmnt_mat = INS.get_atitude_incrment_mat(gyro_vectr)
        return np.eye(3) + \
            ((1 - atitude_incrmnt_norm**2 / 6) * atitude_incrmnt_mat) + \
            ((1 - atitude_incrmnt_norm**2 / 24) * np.matmul(atitude_incrmnt_mat, atitude_incrmnt_mat))
    

    @staticmethod
    def get_skew_sym_mat(vector):
        '''
        Return the skew symmetric matrix of a vector
        '''
        if isinstance(vector, np.ndarray):
            return np.array([[0, -vector.item(2), vector.item(1)],
                            [vector.item(2), 0, -vector.item(0)],
                            [-vector.item(1), vector.item(0), 0]])
        else:
            return np.array([[0, -vector[2], vector[1]], 
                            [vector[2], 0, -vector[0]], 
                            [-vector[1], vector[0], 0]])
    
    @staticmethod
    def get_unskewed(mat):
        '''
        Return the vector corresponding to a skew symmetric matrix
        '''
        return np.array([mat[2,1], mat[0,2], mat[1,0]])



    @staticmethod
    def get_attitude_frm_coord_trnsfrm(coord_trnsfrm_mat: np.ndarray) -> np.ndarray:
        '''
        Calcuate Euler Angles given a coordiante transformation matrix
        '''
        roll = atan2(coord_trnsfrm_mat[1,2],coord_trnsfrm_mat[2,2])
        pitch = -asin(coord_trnsfrm_mat[0,2])
        yaw = atan2(coord_trnsfrm_mat[0,1], coord_trnsfrm_mat[0,0])
        return np.array([roll, pitch, yaw])



    @staticmethod
    def get_coord_transfrm_frm_atitude(euler_vectr: np.ndarray) -> np.ndarray:
        '''
        Calculate coordinate transformation matrix from current euelr angle
        '''
        roll = euler_vectr[0]
        pitch = euler_vectr[1]
        yaw = euler_vectr[2]
        return np.array([
                    [cos(pitch)*cos(yaw),  
                    (-cos(roll)*sin(yaw) + sin(roll)*sin(pitch)*cos(yaw)), 
                    (sin(roll)*sin(yaw) + cos(roll)*sin(pitch)*cos(yaw))
                    ],
                    [cos(pitch)*sin(yaw),  
                    (cos(roll)*cos(yaw) + sin(roll)*sin(pitch)*sin(yaw)), 
                    (-sin(roll)*cos(yaw) + cos(roll)*sin(pitch)*sin(yaw))
                    ],
                    [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]]).T
    
    @staticmethod
    def get_local_earth_rot(lat_rad: float) -> np.ndarray:
        return INS.EATH_ROTN_RATE * INS.get_skew_sym_mat(np.array([cos(lat_rad), 0, -sin(lat_rad)]))
    
    @staticmethod
    def get_local_trnsprt_rate(height:float, lat_rad:float, vel_n:float, vel_e:float)->np.ndarray:
        return INS.get_skew_sym_mat(
                np.array([
                    vel_e / (INS.RADIUS_TRNVRS * lat_rad + height),
                    -vel_n / (INS.RADIUS_MRIDNL * lat_rad + height),
                    -vel_e * tan(lat_rad) / (INS.RADIUS_TRNVRS * lat_rad + height)
                            ])
                    )
    
    @staticmethod
    def get_atitude_incrmnt_vectr(gyro_vectr:np.ndarray)->np.ndarray:
        return INS.TIME_INTRVL * gyro_vectr

    @staticmethod
    def get_atitude_incrment_mat(gyro_vectr:np.ndarray)->np.ndarray:
        return INS.TIME_INTRVL * INS.get_skew_sym_mat(gyro_vectr)
    


    @classmethod
    def setMeridianCurvRadius(cls, lat_fix, long_fix):
        cls.INS.RADIUS_MRIDNL = cls.getMeridianCurvRadius(lat_fix, long_fix)

    @classmethod
    def setTransverseCurvRadius(cls, lat_fix, long_fix):
        cls.RADIUS_PLR = cls.getTransverseCurvRadius(lat_fix, long_fix)

    @classmethod
    def setTimeInterval(cls, t):
        cls.TIME_INTRVL = t

    @classmethod 
    def clearMeridianCurvRadius(cls):
        cls.INS.RADIUS_MRIDNL = False

    @classmethod 
    def clearTransverseCurvRadius(cls):
        cls.RADIUS_PLR = False

    @classmethod
    def clearTimeInterval(cls):
        cls.TIME_INTRVL = False

    