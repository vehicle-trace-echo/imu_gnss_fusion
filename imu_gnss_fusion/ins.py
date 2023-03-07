from math import sin, cos, tan, sqrt, asin, atan2, degrees, pi
from math import radians as rad
import numpy as np


class INS():
    '''
    Inertial Navigation System (INS) mechanizes the IMU in the Local Navigation Frame.

    INS works in parallel with and ES-EKF, for Atitude, velocity, and Position corrections, and
    IMU biases.

    The Mechanization is based on formulas found in 'Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems' - Paul D Groves.
    '''

    RADIUS_EQTRL = 6378137.0            # Equitorial Radius of Ellipsoid (WGS84)
    RADIUS_PLR = 6356752.3142           # Polar Radius of Ellipsoid      (WGS84)
    FLTNING = 1 / 298.257223563         # Flattening                     (WGS84)
    ECNTRCTY = 0.0818181908425          # Eccentricity                   (WGS84)
    EATH_ROTN_RATE = 7.292115E-5        # Angular rotation rate of earth in rad/s
    RADIUS_MRIDNL = False               # RN
    RADIUS_TRNVRS = False               # RE  
    TIME_INTRVL = False   
    LOCAL_GRAV_MAGNTD = 9.8066          
    LOCAL_GRAV_VECTR = np.array([0, 0, LOCAL_GRAV_MAGNTD])




    def __init__(self, init_atitude: np.ndarray,
            init_velocity: np.ndarray,
            init_position: np.ndarray,
            init_bias_accel: np.ndarray,
            init_bias_gyro:np.ndarray,
            sample_rate: int):
        '''
        Initialize the INS in Local Navigation Frame.
        Meridional and Transverse radii of curvature approx constant after the init position fix.

            :param @init_atitude         
                    <brief>             initial atitude (Euler Angles wrt. Local Nav Frame)
                    <origin>            Pitch and Roll from Leveling process, Yaw from aligned GNSS recievers.              
                    <order>             Roll, Pitch, Yaw
                    <units>             Radians

            :param @init_velocity       
                    <brief>             intial velocity wrt. Local Navigation frame
                    <origin>            initial velocity should be zeros; Calibration then Leveling precede INS init       
                    <order>             VelocityNorth, VelocityEast,  VelocityDown 
                    <units>             Meters Per Seconds

            :param @init_position       
                    <brief>             initial position in Geographic Coordinate System 
                    <origin>            GNSS static position during Calibration and Leveling
                    <order>             Ellipsoidal Height, Latitude, Longitude
                    <units>             Meters, Decimal Degreees, Decimal Degrees

            :param @init_bias_accel     
                    <brief>             accelerometer biases
                    <origin>            IMU Calibration
                    <order>             bias_ax, bias_ay, bias_az (x, y, z of body frame)
                    <units>             Meters per squared Seconds

            :param @init_gyro_accel     
                    <brief>             Gyroscope biases
                    <origin>            IMU Calibration
                    <order>             bias_gx, bias_gy, bias_gz (x, y, z of body frame)
                    <units>             Radians per Seconds

            :param @sample_rate
                    <brief>             IMU data rate (hz)
                    <origin>            
                    <order>             
                    <units>             
                    <notes>             Might make this class variable at 100 hz
        '''
        INS.setMeridianCurvRadius(init_position[1], init_position[2])
        INS.setTransverseCurvRadius(init_position[1], init_position[2])
        INS.setTimeInterval(1/sample_rate)

        self.atitude = init_atitude                                                 # Current Navigation solutions
        self.coord_trnsfrm = INS.get_coord_transfrm_frm_atitude(init_atitude)       # Euler angles to 
        self.velocity = init_velocity
        self.position = init_position
        self.bias_accel = init_bias_accel                               # Current IMU biases
        self.bias_gyro = init_bias_gyro

            


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  MAIN APIs   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def INS_UpdateStates(self,
            acc_vectr:np.ndarray,
            gyro_vectr:np.ndarray):
        ''' 
        Upate INS states on IMU data ready
            :param  @gyro_vectr        
                <brief>             Vector containing latest accelerometer readings
                <origin>            IMU            
                <order>             ax, ay, az  (Body Frame)
                <units>             Meters per sqaured Seconds

            :param  @gyro_vectr        
                <brief>             Vector containing latest gyroscope readings
                <origin>            IMU            
                <order>             gx, gy, gz  (Body Frame)
                <units>             Radians per Seconds
        '''
        # Some parameters reused across updates
        self.lat_rad = rad(self.position[1])                                    # Latitude in Radians                        
        self.earth_rot_mat = INS.get_local_earth_rot(self.lat_rad)                # Earth rotation rate in Local Nav Frame
                                                                                # And
        self.trnsprt_mat = INS.get_local_trnsprt_rate(self.position[0], self.lat_rad,  # Local Navigation Frame transport rate        
                                          self.velocity[0], self.velocity[1])   # based on current soltuions
        # Update States
        self.update_atitude(acc_vectr, gyro_vectr)
        self.update_velocity(acc_vectr, gyro_vectr)
        self.update_position()
        self.INS_ResetNexts()
        


    def INS_ResetNexts(self):
        '''INS_Update tracks current and next solutions;
        update current to next, and reset next'''

        self.atitude = self.next_atitude                                                           
        self.coord_trnsfrm = self.next_coord_trnsfrm
        self.velocity = self.next_velocity
        self.position = self.next_position
        # del self.next_atitude, self.next_velocity, self.next_position, self.next_coord_trnsfrm



    def INS_Correct(self, atitude_crrctn:np.ndarray,
                    velocity_crrctn: np.ndarray,
                    position_crrctn: np.ndarray):
         '''
         Apply corrections to INS solutions based on solution from ES-EKF

            :param  @atitude_crrctn        
                <brief>             Vector atitude correction
                <origin>            ES-EKF            
                <order>             delRoll, delPitch, delYaw
                <units>             Radians
                <notes>             'del' represents the variational operator

            :param  @velocity_crrctn        
                <brief>             Vector velocity correction
                <origin>            ES-EKF            
                <order>             delVelocityNorth, delVelocityEast, delVelocityDown
                <units>             Meters per Seconds
                <notes>             'del' represents the variational operator

            :param  @position_crrctn        
                <brief>             Vector position correction
                <origin>            ES-EKF            
                <order>             delHeight, delLatitude, delLongitude
                <units>             Meters, Decimal Degreees, Decimal Degrees
                <notes>             'del' represents the variational operator
         '''
         self.coord_trnsfrm = np.matmul(                                                            # Apply atitude crrctn (small angle approx)
                                (np.eye(3) - INS.get_skew_sym_mat(atitude_crrctn)),                 # to coord trnsfrm soln
                                    self.coord_trnsfrm)
         self.atitude = INS.get_attitude_frm_coord_trnsfrm(self.next_coord_trnsfrm)                 # crrctd atitude from crrted coord trnsfrm  
         
         self.velocity = self.velocity - velocity_crrctn
         self.position = self.position - position_crrctn




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Atitude Update  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_atitude(self,
            acc_vectr:np.ndarray,
            gyro_vectr:np.ndarray,
            comp_filter=None, 
            )->np.ndarray:
        '''
        <brief>         Update Atitude for INS based on latest IMU readings.

        <background>    Orientation of INS tracked with Euler Angles (self.atitude vector).
                        INS math uses coordinate transform (rotation) matrix (self.coord_trnsfrm) based on Euler angles.
                        Gyroscope readings in body frame (vehicle's) while solutions tracked in Local Nav Frame (NED).
                        Conversion of gyro readings to changes in Euler Angles Frame through Diff. Eqn (DE). 
                        DE in terms of the matrix; soln for which is first approxed, then atitude is extracted.

        <notes>         Need both vector and matrix as INS math relies on matrix, while corrections for ESEKF in terms of 
                        vector. 
        '''
        self.next_coord_trnsfrm = self.updt_coord_trnsfrm(gyro_vectr)                               # Get next coodinate transform matrix

        self.next_atitude = INS.get_attitude_frm_coord_trnsfrm(self.next_coord_trnsfrm)             # crrctd atitude from crrted coord trnsfrm        
        
    

    def updt_coord_trnsfrm(self,
            gyro_vectr:np.ndarray):
        
        ''' 
        Returns the next orientation matrix based on gyroscope readings and current orientation
        Solution  based on a fourth-order approx. to ODE relating Euler Rates to Body Rates.
        '''
        atitude_updt_mat = self.get_atitude_updt_mat(gyro_vectr)

        return np.matmul(self.coord_trnsfrm, atitude_updt_mat) - \
                (np.matmul((self.earth_rot_mat+self.trnsprt_mat), self.coord_trnsfrm) * INS.TIME_INTRVL)



    def get_atitude_updt_mat(self, gyro_vectr:np.ndarray)->np.ndarray:
        '''
        <brief>             Matrix converting current orientation matrix to the next orientation matrix.

        <background>        Intuitive to visualze solving atitude update DE as applying a rotation from last solution to 
                            the next. 
                            This rotation matrix is calculated here.

        '''
        self.atitude_incrmnt_mat = INS.get_atitude_incrment_mat(gyro_vectr)  
                 
        self.atitude_incrmnt_norm =  np.linalg.norm(INS.get_atitude_incrmnt_vectr(gyro_vectr))

        return np.eye(3) + \
            ((1 - self.atitude_incrmnt_norm**2 / 6) * self.atitude_incrmnt_mat) + \
            ((1 - self.atitude_incrmnt_norm**2 / 24) * np.matmul(self.atitude_incrmnt_mat, self.atitude_incrmnt_mat))
    



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Velocity Update  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_velocity(self, 
            accel_vectr: np.ndarray,
            gyro_vectr: np.ndarray
            ):
        '''
        <brief>             Update velocity based on last solutions and IMU update

        <background>        Velocity of Body in Local Navigation Frame is with respect to Earth Centered Earth Fixed (ECEF) 
                            (Local Navigation and Body frame coincide, hence velocity body wrt local nav. is zero).

                            IMU reads Specifc Force (Acceleration - Gravity) in body frame.
                            Specific Force in Navigation Frame required; it is related to Body Frame specific force and 
                            instantaneous Coordinate Transformation Matrix.
                            Instantaenous Coordiante Transfrom cannot be achieved hence, and average across INS.TIME_INTRVL 
                            is approxed. 

                            Using current orientation, gravity is resolved in body frame.

                            Component of Gravity in the forward direction (x) is taken away form IMU reading to get only acceleration.

        <notes>             Only acceleration in the x-axis is assumed to cause changes in velocity (kinematic constraints of cars)
        
        '''
        grav_vectr_body = self.get_grav_vectr_body(gyro_vectr)
        modf_accel_vectr = np.array([accel_vectr[0]+grav_vectr_body[0],0, 0])           # Ax with Gravitational effect removed

        accel_nav = self.get_spcfc_force_nav(modf_accel_vectr, gyro_vectr)              # Acceleration in Local Nav
                                                                                        # Calculate next veloctioy
        self.next_velocity = self.velocity +  \
                                (accel_nav  - \
                                np.matmul(
                                    (self.trnsprt_mat + 2 * self.earth_rot_mat), 
                                    self.velocity)) * INS.TIME_INTRVL
    

    def get_grav_vectr_body(self,
                        gyro_vectr:np.ndarray):
        '''
        Resolve gravity vector into body frame using the average 
        coordinate transform matrix (nav to body) over INS.TIME_INTRVAL
        '''
        return np.matmul(self.get_avg_coord_trnsfrm(
                                    self.coord_trnsfrm.T,
                                    gyro_vectr
                            ), INS.LOCAL_GRAV_VECTR
                         )


    def get_spcfc_force_nav(self, 
            accel_vectr: np.ndarray,
            gyro_vectr:np.ndarray):
        '''
        Convert IMU read accelerometer readings from Body to Local Nav. Frame using 
        '''
        return np.matmul(self.get_avg_coord_trnsfrm(
                                self.coord_trnsfrm,
                                gyro_vectr), 
                            accel_vectr)



    def get_avg_coord_trnsfrm(self,
                coord_trnsfrm:np.ndarray,
                gyro_vectr:np.ndarray):
        '''
        <brief>             Integral average (4th order-approx) of instantaneous coordinate transformation matrix for a INS.TIME_INTRVL
        '''
        return np.matmul(
                        coord_trnsfrm, 
                        self.get_avg_atitude_updt_mat(gyro_vectr)
                        )               \
                    -                   \
                0.5 * np.matmul(
                    (self.earth_rot_mat + self.trnsprt_mat), 
                    coord_trnsfrm
                    ) *  INS.TIME_INTRVL



    def get_avg_atitude_updt_mat(self, gyro_vectr:np.ndarray)->np.ndarray:
        '''
        Fourth Order apprximation of average transformation matrix 
        '''
        return np.eye(3) + \
            0.0416667 * ((12 - self.atitude_incrmnt_norm**2) * self.atitude_incrmnt_mat) + \
            0.008333333 * ((1 - self.atitude_incrmnt_norm**2) * np.matmul(self.atitude_incrmnt_mat, self.atitude_incrmnt_mat))



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Position Update  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_position(self):

        next_height = INS.update_height(self.position[0], self.velocity[2], self.next_velocity[2])             # height, vel_D, next_vel_D

        next_lat = INS.update_latitude(self.position[0], next_height, self.position[1],                        # height, next_height, latitude,
                                        self.velocity[0], self.next_velocity[0])                               # vel_N, next_vel_N
        
        next_long = INS.update_longitude(self.position[0], next_height,                                        # height, next_height, 
                                          self.position[1], next_lat,                                          # latitude, next_latitude
                                          self.position[2],                                                     # longitude 
                                        self.velocity[1], self.next_velocity[1])                               # vel_E, next_vel_E 

        self.next_position =  np.array([next_height, next_lat, next_long])
        


    @staticmethod
    def update_height(height, vel_d, next_vel_d):
        ''' Return next height solution'''
        return height - INS.TIME_INTRVL * (vel_d + next_vel_d)
    


    @staticmethod
    def update_latitude(height, next_height, lat, vel_n, next_vel_n):
        ''' Return next latitude soltuion'''
        return lat + degrees(INS.TIME_INTRVL / 2 * (\
                    vel_n/(INS.RADIUS_MRIDNL  + height) + \
                    next_vel_n/(INS.RADIUS_MRIDNL  + next_height)
            ))


    @staticmethod
    def update_longitude(height, next_height, lat, next_lat, long, vel_e, next_vel_e):
        ''' Return next logitude soltuon '''
        return long + degrees(INS.TIME_INTRVL / 2 * (\
                vel_e/((INS.RADIUS_TRNVRS+ height) * cos(rad(lat))) + \
                next_vel_e/((INS.RADIUS_TRNVRS+ + next_height) * cos(rad(next_lat)))
            ))



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Math Preliminaries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def heading_to_yaw(heading_rads):
        if heading_rads  < pi:
            return heading_rads
        else:
            return -1* (2*pi - heading_rads)

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
        roll = atan2(coord_trnsfrm_mat[2,1],coord_trnsfrm_mat[2,2])
        pitch = -asin(coord_trnsfrm_mat[2,0])
        yaw = atan2(coord_trnsfrm_mat[1,0], coord_trnsfrm_mat[0,0])
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
                    [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]])
    


    @staticmethod
    def cartsn_to_curvilin_pos(posn_vectr,
                               lat_rad, height):
        return np.matmul(np.array([
                                [1/(INS.RADIUS_MRIDNL + height), 0, 0],
                                [0,     1/((INS.RADIUS_TRNVRS + height)*cos(lat_rad)),0]
                                [0,     0,  -1]]), posn_vectr)



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Misc.  Functions   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def get_atitude_incrmnt_vectr(gyro_vectr:np.ndarray)->np.ndarray:
        '''
        <brief>             Return, as a vector, increment in atitude in body frame over INS.TIME_INTRVL 
                            from gyro reading .

        <notes>             Gyroscope readings are assumed to be constant through INS.TIME_INTRVL
        '''
        return INS.TIME_INTRVL * gyro_vectr



    @staticmethod
    def get_atitude_incrment_mat(gyro_vectr:np.ndarray)->np.ndarray:
        '''
        <brief>             Return, as a skew-symmetric matrix, increment in atitude in body frame over 
                            INS.TIME_INTRVL from gyro reading .

        <notes>             Gyroscope readings are assumed to be constant through INS.TIME_INTRVL
        '''
        return INS.TIME_INTRVL * INS.get_skew_sym_mat(gyro_vectr)
    


    @staticmethod
    def get_local_earth_rot(lat_rad: float) -> np.ndarray:
        '''
        <brief>             returns the rotation speed of the earth about its axis resolved in the 
                            local navigation frame
        '''
        return INS.EATH_ROTN_RATE * INS.get_skew_sym_mat(np.array([cos(lat_rad), 0, -sin(lat_rad)]))
    


    @staticmethod
    def get_local_trnsprt_rate(height:float, lat_rad:float, vel_n:float, vel_e:float)->np.ndarray:
        '''
        <breif>             returns rotation of the local navigation frame wrt to the earth centre
        '''
        return INS.get_skew_sym_mat(
                np.array([
                    vel_e / (INS.RADIUS_TRNVRS * lat_rad + height),
                    -vel_n / (INS.RADIUS_MRIDNL * lat_rad + height),
                    -vel_e * tan(lat_rad) / (INS.RADIUS_TRNVRS * lat_rad + height)
                            ])
                    )




    @staticmethod
    def getMeridianCurvRadius(lat_fix, long_fix):
        '''
        Return R_n (Meridional Raidus of Curvature)
        '''
        return INS.RADIUS_EQTRL * (1-INS.ECNTRCTY**2) / \
            (1 - INS.ECNTRCTY**2 *                    \
            (sin(rad(lat_fix)))**2) ** (3/2)




    @staticmethod
    def getTransverseCurvRadius(lat_fix, long_fix):
        '''
        Return R_e (Transverse Radius of Curvature)
        '''
        return INS.RADIUS_EQTRL / sqrt(1-INS.ECNTRCTY**2 * \
                (sin(rad(lat_fix)))**2)



    @classmethod
    def setMeridianCurvRadius(cls, lat_fix, long_fix):
        cls.RADIUS_MRIDNL = cls.getMeridianCurvRadius(lat_fix, long_fix)



    @classmethod
    def setTransverseCurvRadius(cls, lat_fix, long_fix):
        cls.RADIUS_TRNVRS = cls.getTransverseCurvRadius(lat_fix, long_fix)



    @classmethod
    def setTimeInterval(cls, t):
        cls.TIME_INTRVL = t



    @classmethod 
    def clearMeridianCurvRadius(cls):
        cls.INS.RADIUS_MRIDNL = False



    @classmethod 
    def clearTransverseCurvRadius(cls):
        cls.RADIUS_TRNVRS = False



    @classmethod
    def clearTimeInterval(cls):
        cls.TIME_INTRVL = False

    