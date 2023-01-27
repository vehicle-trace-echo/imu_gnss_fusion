import math
import numpy as np


class KalmanFilter():
    '''
    Linear Kalman Filter algorithm.
    @Notes: 
    '''
    def __init__(self,
                MAT_STATE_TRANSTN: np.array,
                MAT_CNTRL: np.array,
                MAT_OBSRVN: np.array,
                MAT_PROCESS_NOISE: np.array,
                MAT_MEAS_UNCERTT: np.array):
        '''
        Initialize the Kalman filter based on the minimum required parameters.
        
        :param @MAT_STATE_TRANSTN:          state transition matrix; relates current state to next during PREDICT
        :param @MAT_CNTRL:                  control matrix; realates current inputs to next state during PREDICT
        :param @MAT_OBSRVN:                 observation matrix; relates measurements (observations) to state variables during UPDATE                                    
        :param @MAT_PROCESS_NOISE:          process noise; certainty in control inputs and process model
        :param @MAT_MEAS_UNCERT:            measurement noise; certainty in observations 
        '''
        self.set_state_transition_mat(MAT_STATE_TRANSTN)    
        self.set_control_mat(MAT_CNTRL)
        self.set_observation_mat(MAT_CNTRL)

    

    # Setter Method
    def set_state_transition_mat(self, MAT_STATE_TRANSTN : np.matrix) -> None:
        '''
        Extract # of rows (@_ROWS_STATE_VECTR).
        Check for sqaure matrix.
        Set @_MAT_STATE_TRANSTN.

        :param @MAT_STATE_TRANSTN:          @MAT_STATE_TRANSTN
        '''
        _n = MAT_STATE_TRANSTN.shape[0]
        _m = MAT_STATE_TRANSTN.shape[1]

        if(_n == _m):
            self._MAT_STATE_TRANSTN = MAT_STATE_TRANSTN
            self._ROWS_STATE_VECTR = _n
            return
        raise InvalidMatrixDimensions(f"State Transtion (F) must be a square matrix. Current dimensions: ({_n} x {_m})")
    

    # Setter Method 
    def set_control_mat(self, MAT_CNTRL: np.matrix) -> None:
        '''
        Extract # number of rows (@_ROWS_CNTRL_VECTR) 
        Check for matrix dimensional consistency with @_MAT_STATE_TRANSTN.
        Set @_MAT_CNTRL.

        :param @MAT_CNTRL:                  @MAT_CNTRL
        '''
        _n = MAT_CNTRL.shape[0]
        _m = MAT_CNTRL.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Control Matrix dimensions ({_n} x {_m}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Control Matrix must have the dimensions ({self._ROWS_STATE_VECTR } x m)")

        self._MAT_CNTRL = MAT_CNTRL
        self._ROWS_CNTRL_VECTR = _m


    # Setter Method
    def set_observation_mat(self, MAT_OBSRVN: np.matrix) -> None:
        '''
        Extract # number of rows (@_ROWS_MEAS_VECTR)
        Check for matrix dimensional consistency with @_MAT_STATE_TRANSTN
        Set @_MAT_OBSRVN

        :param @MAT_OBSRVN:                 @MAT_OBSRVN
        '''
        _k = MAT_OBSRVN.shape[0]
        _n = MAT_OBSRVN.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Observation Matrix dimensions ({_k} x {_n}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Observation Matrix must have the dimensions (k x {self._ROWS_STATE_VECTR })")

        self._MAT_OBSRVN = MAT_OBSRVN
        self._ROWS_OBSRVN_VECTR = _k





class InvalidMatrixDimensions(Exception):
    '''
    Matrix dimensions not as expected. E.g. F must be square.
    '''
    pass

class InconsistentMatrixDimensions(Exception):
    '''
    Matrix dimensions are inconsistent with other key matrix.

    E.g. if F is n x n   and G  is i x j, then n must equal i. 
    '''
    pass