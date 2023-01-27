import math
import numpy as np


class KalmanFilter():
    '''
    Linear Kalman Filter algorithm.
    @Notes: 
    '''
    def __init__(self,
                STATE_TRANSTN_MAT: np.array,
                CONTROL_MAT: np.array,
                OBSRVN_MAT: np.array,
                PROCESS_NOISE: np.array,
                MEAS_UNCERT: np.array):
        '''
        Initialize the Kalman filter based on the minimum required parameters.
        
        :param @STATE_TRANSTN_MAT:      state transition matrix; relates current state to next during PREDICT
        :param @CONTROL_MAT:            control matrix; realates current inputs to next state during PREDICT
        :param @OBSRVN_MAT:             observation matrix; relates measurements (observations) to state variables during UPDATE                                    
        :param @PROCESS_NOISE:          process noise; certainty in control inputs and process model
        :param @MEAS_UNCER:             measurement noise; certainty in observations 
        '''
        self.set_state_transition(STATE_TRANSTN_MAT)    

    # setter method
    def set_state_transition(self, STATE_TRANSTN_MAT : np.array) -> None:
        '''
        Set the state transition matrix, extract the number of rows (n)
        in STATE VECTOR to check for overall matrix dimension consistency
        '''
        _n = STATE_TRANSTN_MAT.shape[0]
        _m = STATE_TRANSTN_MAT.shape[1]

        if(_n == _m):
            self._STATE_TRANSTN_MAT = STATE_TRANSTN_MAT
            self.rows_state_vector = _n
            return
        raise InvalidMatrixDimension("State Transtion (F) must be a square matrix")





class InvalidMatrixDimension(Exception):
    pass

