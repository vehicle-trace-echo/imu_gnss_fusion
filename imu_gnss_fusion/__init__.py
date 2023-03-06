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
            sample_rate: int):
        self.ins = INS(init_atitude,
                       init_velocity,
                       init_position,
                       init_bias_accel,
                       init_bias_gyro,
                       sample_rate)
        self.atitude_crrctn = np.zeros(3)
        self.velocity_crrctn = np.zeros(3)
        self.position_crrctn = np.zeros(3)
        self.bias_accel = init_bias_accel
        self.bias_gyro = init_bias_gyro



    def EsEKF_StateTransition(self,
                              raw_accel_vectr: np.ndarray,
                              raw_gyro_vectr:np.ndarray):
        
        accel_vectr = IMU.calibrate_accel(IMU.normalize_accel(raw_accel_vectr))
        gyro_vectr = IMU.calibrate_gyro(IMU.normalize_gyro(raw_gyro_vectr))

        self.ins.INS_UpdateStates(accel_vectr, gyro_vectr)

        self.ins.INS_Correct(self.atitude_crrctn, 
                            self.velocity_crrctn,
                            self.position_crrctn
                             )

        pass



    def get_state_trnstn_mdl(self,
                             accel_vectr: np.ndarray,
                             gyro_vectr: np.ndarray)->np.ndarray:
        
        F11 = -1 * INS.get_skew_sym_mat(gyro_vectr)

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
        F21 = -1 * INS.get_skew_sym_mat(np.matmul(self.ins.coord_trnsfrm, accel_vectr))

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
        
        F = np.eye(15)
        F[:3, :3] = F[:3, :3] + F11 * self.ins.TIME_INTRVL
        F[:3, 3:6] = F12 * self.ins.TIME_INTRVL
        F[:3, 6:9] = F13 * self.ins.TIME_INTRVL
        F[:3, 12:] = self.ins.coord_trnsfrm * self.ins.TIME_INTRVL

        F[3:6, :3] =  F21 * self.ins.TIME_INTRVL
        F[3:6, 3:6] = F[3:6, 3:6] + F22 * self.ins.TIME_INTRVL
        F[3:6, 6:9] = F23 * self.ins.TIME_INTRVL
        F[9:12, 9:12] = self.ins.coord_trnsfrm * self.ins.TIME_INTRVL

        F[6:9, 3:6] = F[6:9, 3:6] * self.ins.TIME_INTRVL
        F[6:9, 6:9] = F[6:9, 6:9] + F33 * self.ins.TIME_INTRVL

        return F





