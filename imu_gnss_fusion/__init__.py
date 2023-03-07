from math import sin, cos, tan, sec, sqrt, asin, atan2, degrees, pi
from math import radians as rad
from ins import INS
from imu import IMU
import numpy as np



class ImuGnssEsEKF():


    def __init__(self,
                init_atitude: np.ndarray,
                init_velocity: np.ndarray,
                init_position: np.ndarray,
                init_bias_accel: np.ndarray,
                init_bias_gyro:np.ndarray,
                procss_noise: np.ndarray,
                measmnt_uncrt: np.ndarray,
                lever_arm: np.ndarray,
                sample_rate: int):
        
        self.procss_noise = procss_noise
        self.measmnt_uncrt = measmnt_uncrt

        self.init_state_vectr(init_bias_accel,
                       init_bias_gyro)
        self.init_covar_mat()
        self.init_measmnt_mat()
        

        self.ins = INS(init_atitude,
                       init_velocity,
                       init_position,
                       init_bias_accel,
                       init_bias_gyro,
                       sample_rate,)
        
        self.lever_arm = lever_arm
        


    def EsEKF_Predict(self,
                    raw_accel_vectr: np.ndarray,
                    raw_gyro_vectr:np.ndarray):
        
        self.ins.accel_vectr = IMU.calibrate_accel(IMU.normalize_accel(raw_accel_vectr))
        self.ins.gyro_vectr = IMU.calibrate_gyro(IMU.normalize_gyro(raw_gyro_vectr))

        self.ins.INS_UpdateStates()

        self.ins.INS_Correct(self.atitude_crrctn, 
                            self.velocity_crrctn,
                            self.position_crrctn) 
             

    
    def EsEKF_extrapolate_state(self):
        self.update_state_trnstn_mat(self.ins.accel_vectr, self.ins.gyro_vectr)
        self.state_vectr = np.matmul(self.state_trnsntn_mat, self.state_vectr) 

    

    def EsEKF_extrapolate_covar(self):
        self.covar_mat = np.matmul(np.matmul(self.state_trnsntn_matself,
                                             self.covar_mat),
                                    self.state_trnsntn_mat
                                    ) + self.procss_noise
        
            
    def EsEFK_Update(self,
                measmnt_vectr: np.ndarray,
                measmnt_uncrt_mat: np.ndarray):
        
        self.EsEKF_compute_k_gain(measmnt_uncrt_mat)
        self.EsEKF_compute_innovatn(measmnt_vectr)
        self.EsEKF_update_state()
        self.EsEKF_update_covar()
    


    def EsEKF_compute_k_gain(self, measmnt_uncrt_mat:np.ndarray):

        PH = np.matmul(self.measmnt_mat, self.covar_mat)
        HPH = np.matmul(PH, self.measmnt_mat)
        self.kalman_gain = np.matmul(PH.T, 
                                np.linalg.pinv(
                                        HPH.T + measmnt_uncrt_mat
                                        )
                                        )
        
    

    def EsEKF_compute_innovatn(self, measmnt_vectr:np.ndarray):
        self.velocity_innovatn = measmnt_vectr[:3] - self.ins.velocity \
                            - np.matmul(self.ins.coord_trnsfrm, 
                                        np.linalg.cross(self.ins.gyro_vectr,
                                                        self.lever_arm)) \
                            + np.matmul(
                                    np.matmul((self.ins.earth_rot_mat + self.ins.trnsprt_mat), 
                                        np.matmul(self.ins.coord_trnsfrm,
                                            self.lever_arm)
                                    )
                            )
        
        self.position_innovatn = measmnt_vectr[3:] - self.ins.position \
                    - self.ins.cartsn_to_curvilin_trnsfrm(
                        np.matmul(self.ins.coord_trnsfrm, 
                                    self.lever_arm))
        
        self.innovatn_vectr = np.append(self.velocity_innovatn, self.position_innovatn)



    def EsEKF_update_state(self):
        self.state_vectr = self.state_vectr + \
                            np.matmul(self.kalman_gain,
                                      self.innovatn_vectr)
        

    
    def EsEKF_update_covar(self):
        self.covar_mat = np.matmul(
                        (np.eye(15) - np.matmul(self.kalman_gain, 
                                                self.measmnt_mat)),
                        self.covar_mat
                    )



    def update_state_trnstn_mat(self)->np.ndarray:
        
        F11 = -1 * INS.get_skew_sym_mat(self.ins.gyro_vectr)

        F12 = np.array([[0,     
                    -1/(self.ins.RADIUS_TRNVRS  + self.ins.position[0]),     
                    0
                    ],
                    [1 / (self.ins.RADIUS_MRIDNL  + self.ins.position[0]),    
                    0,
                    0
                    ],
                    [0,     
                    tan(self.ins.lat_rad) / (self.ins.RADIUS_TRNVRS + self.ins.position[0]),  
                    0
                    ]
                ])
        
        F13 = np.array([
                [self.ins.EATH_ROTN_RATE * sin(self.ins.lat_rad),   
                0,
                self.ins.velocity[1] * F12[0,1]**2
                ],
                [0,
                0,
                -1 * self.ins.velocity[0] * F12[1,0]**2
                ],          
                [self.ins.EATH_ROTN_RATE * cos(self.ins.lat_rad) \
                 + F12[0,1] * self.ins.velocity[1] / cos(self.ins.lat_rad)**2, 
                0,
                self.ins.velocity[1]  *  F12[2,1] * F12[0,1]
                ]
            ])
        # Car Kinematic constraint not applied!
        F21 = -1 * INS.get_skew_sym_mat(np.matmul(self.ins.coord_trnsfrm, self.ins.accel_vectr))

        F22 =  np.array([
                [self.ins.velocity[2]/F12[1,0],    
                -2*self.ins.velocity[1] * F12[2,1] - 2*F13[0,0],
                self.ins.velocity[0] * F12[1,0]
                ],
                [self.ins.velocity[1] * F12[2,1] + 2 * F13[0,0],
                self.ins.velocity[0] * F12[2,1] + self.ins.velocity[2] * F12[2,1],
                -1 * self.ins.velocity[1] * F12[0,1] + 2 * self.ins.EATH_ROTN_RATE * cos(self.ins.lat_rad),
                ],
                [-2*self.ins.velocity[0] * F12[1,0], 
                 2*self.ins.velocity[1] * F12[0,1] - 2 * self.ins.EATH_ROTN_RATE * cos(self.ins.lat_rad),
                0
                 ]
            ])
        F23 = np.array([
                [self.ins.velocity[1]**2 * sec(self.ins.lat_rad)**2 * F12[0,1] \
                 - 2 * self.ins.velocity[1] *  self.ins.EATH_ROTN_RATE * cos(self.ins.lat_rad),
                0,
                F13[2,2] * -1 * self.ins.velocity[1] + F13[1,2] * self.ins.velocity[2]            
                ],
                [self.ins.velocity[0]* self.ins.velocity[1] * sec(self.ins.lat_rad)**2 * F12[0,1] \
                 + 2 * self.ins.velocity[0] *  self.ins.EATH_ROTN_RATE * cos(self.ins.lat_rad) \
                - 2 * self.ins.velocity[2] * F13[0,0],
                0,
                F13[2,2] * self.ins.velocity[0] - F13[0,2] * self.ins.velocity[2]
                ],
                [2 * self.ins.velocity[1] * F13[0,0],
                 0,
                 F13[0,2] * self.ins.velocity[1] + F13[1,2] * -1 * self.ins.velocity[0]            
                ]
            ])
        
        F32 = np.array([
                [F12[1,0],  
                0,
                0
                ],
                [0,
                -1* F12[0,1] / cos(self.ins.lat_rad),
                0
                ],
                [0,
                 0,
                -1
                ]
            ])
        
        F33 = np.array([
                    [0,
                     0,
                     F13[1,2]
                     ],
                     [self.ins.velocity[1] * sin(self.ins.lat_rad) * -1 * F12[0,1] / cos(self.ins.lat_rad)**2,
                      0,
                      F13[0,2] / cos(self.ins.lat_rad)
                     ],
                     [0, 
                     0,
                     0]
            ])
        
        self.state_trnsntn_mat = np.eye(15)
        self.state_trnsntn_mat[:3, :3] = self.state_trnsntn_mat[:3, :3] + F11 * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[:3, 3:6] = F12 * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[:3, 6:9] = F13 * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[:3, 12:] = self.ins.coord_trnsfrm * self.ins.TIME_INTRVL

        self.state_trnsntn_mat[3:6, :3] =  F21 * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[3:6, 3:6] = self.state_trnsntn_mat[3:6, 3:6] + F22 * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[3:6, 6:9] = F23 * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[9:12, 9:12] = self.ins.coord_trnsfrm * self.ins.TIME_INTRVL

        self.state_trnsntn_mat[6:9, 3:6] = self.state_trnsntn_mat[6:9, 3:6] * self.ins.TIME_INTRVL
        self.state_trnsntn_mat[6:9, 6:9] = self.state_trnsntn_mat[6:9, 6:9] + F33 * self.ins.TIME_INTRVL



    def init_state_vectr(self,
                        init_bias_accel,
                        init_bias_gyro):
        self.state_vectr = np.zeros(15)
        self.state_vectr[9:12] = init_bias_accel
        self.state_vectr[12:] = init_bias_gyro

    

    def init_covar_mat(self):
        self.covar_mat = np.eye(15)
                


    def init_measmnt_mat(self):
        self.measmnt_mat = np.zeros((6,15))
        self.measmnt_mat[0:3,6:9] = -1 * np.eye(3)
        self.measmnt_mat[3:6,3:6] = -1 * np.eye(3)

    




