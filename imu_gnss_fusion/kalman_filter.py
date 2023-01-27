import math
import numpy as np


class KalmanFilter():
    '''
    Linear Kalman Filter algorithm.
    @Notes: 
    '''
    def __init__(self,
                MAT_STATE_TRANSTN: np.matrix,
                MAT_CNTRL: np.matrix,
                MAT_OBSRVN: np.matrix,
                MAT_PROCESS_NOISE: np.matrix,
                MAT_MEAS_UNCERT: np.matrix,
                VECTR_INIT_STATE: np.array,
                MAT_INIT_COVAR: np.matrix):
        '''
        Initialize the Kalman filter based on the minimum required parameters.
        
        :param @MAT_STATE_TRANSTN:          state transition matrix; relates current state to next during PREDICT
        :param @MAT_CNTRL:                  control matrix; realates current inputs to next state during PREDICT
        :param @MAT_OBSRVN:                 observation matrix; relates measurements (observations) to state variables during UPDATE                                    
        :param @MAT_PROCESS_NOISE:          process noise; uncertainty in control inputs and process model
        :param @MAT_MEAS_UNCERT:            measurement noise; uncertainty in observations 
        :param @VECTR_INIT_STATE:           initial state vector, guessed or provided by a different process
        :param @MAT_INIT_COVAR:             initial covariance matrix; uncertainty in initial state vector
        '''

        self.set_state_transition_mat(MAT_STATE_TRANSTN)    
        self.set_control_mat(MAT_CNTRL)
        self.set_observation_mat(MAT_OBSRVN)
        self.set_process_noise(MAT_PROCESS_NOISE)
        self.set_measurement_uncertainty(MAT_MEAS_UNCERT)
        self.set_initial_state(VECTR_INIT_STATE)
        self.set_initial_covariance(MAT_INIT_COVAR)

    

    # Setter Method
    def set_state_transition_mat(self, MAT_STATE_TRANSTN : np.matrix) -> None:
        '''
        Extract # of rows (@ROWS_STATE_VECTR).
        Check for sqaure matrix.
        Set @MAT_STATE_TRANSTN.

        :param @MAT_STATE_TRANSTN:          @MAT_STATE_TRANSTN
        '''
        _n = MAT_STATE_TRANSTN.shape[0]
        _x = MAT_STATE_TRANSTN.shape[1]

        if(_n == _x):
            self._MAT_STATE_TRANSTN = np.copy(MAT_STATE_TRANSTN)
            self._ROWS_STATE_VECTR = _n
            return
        raise InvalidMatrixDimensions(f"State Transtion must be a square matrix. Current dimensions: ({_n} x {_x})")
    

    # Setter Method 
    def set_control_mat(self, MAT_CNTRL: np.matrix) -> None:
        '''
        Extract # number of rows (@ROWS_CNTRL_VECTR) 
        Check for matrix dimensional consistency with @MAT_STATE_TRANSTN.
        Set @MAT_CNTRL.

        :param @MAT_CNTRL:                  @MAT_CNTRL
        '''
        _n = MAT_CNTRL.shape[0]
        _m = MAT_CNTRL.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Control matrix dimensions ({_n} x {_m}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Control matrix must have the dimensions ({self._ROWS_STATE_VECTR } x m)")

        self._MAT_CNTRL = np.copy(MAT_CNTRL)
        self._ROWS_CNTRL_VECTR = _m


    # Setter Method
    def set_observation_mat(self, MAT_OBSRVN: np.matrix) -> None:
        '''
        Extract # number of rows (@ROWS_MEAS_VECTR)
        Check for matrix dimensional consistency with @MAT_STATE_TRANSTN
        Set @MAT_OBSRVN

        :param @MAT_OBSRVN:                 @MAT_OBSRVN
        '''
        _k = MAT_OBSRVN.shape[0]
        _n = MAT_OBSRVN.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Observation Matrix dimensions ({_k} x {_n}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Observation Matrix must have the dimensions (k x {self._ROWS_STATE_VECTR })")

        self._MAT_OBSRVN = np.copy(MAT_OBSRVN)
        self._ROWS_MEAS_VECTR = _k


    # Setter Method
    def set_process_noise(self, MAT_PROCESS_NOISE: np.matrix) -> None:
        '''
        Extract # number of rows (@ROWS_STATE_VECTOR).
        Check for matrix dimensional consistency with @MAT_STATE_TRANSTN
        Set @MAT_PROCESS_NOISE

        :param @MAT_PROCESS_NOISE:             @MAT_PROCESS_NOISE
        '''
        _n = MAT_PROCESS_NOISE.shape[0]
        _x = MAT_PROCESS_NOISE.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Process Noise Matrix dimensions ({_n} x {_x}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Process Noise Matrix must have the dimensions ({self._ROWS_STATE_VECTR}  x {self._ROWS_STATE_VECTR})")

        if(not _n == _x):
            raise InvalidMatrixDimensions(f"Process Noise must be a square matrix. \
             Current dimensions: ({_n} x {_x})")

        self._MAT_PROCESS_NOSE = np.copy(MAT_PROCESS_NOISE)


    # Setter Method
    def set_measurement_uncertainty(self, MAT_MEAS_UNCERT: np.matrix) -> None:
        '''
        Extract # number of rows (@ROWS_MEAS_VECTR). 
        Check for matrix dimensional consistency with @MAT_OBSRVN
        Set @MAT_MEAS_UNCERT

        :param @MAT_MEAS_UNCERT:                @MAT_MEAS_UNCERT
        '''
        _k = MAT_MEAS_UNCERT.shape[0]
        _x = MAT_MEAS_UNCERT.shape[1]

        if(not _k == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Measurement Uncertainty Matrix dimensions ({_k} x {_x}), inconsistent \
                with observation matrix ({self._ROWS_MEAS_VECTR} x {self._ROWS_MEAS_VECTR}).\n \
                Measurement Uncertainty Matrix must have the dimensions ({self._ROWS_MEAS_VECTR}  x {self._ROWS_MEAS_VECTR})")

        if(not _k == _x):
            raise InvalidMatrixDimensions(f"Measurement Uncertainty must be a square matrix. \
             Current dimensions: ({_k} x {_x})")
            
        self._MAT_MEAS_UNCERT = MAT_MEAS_UNCERT


    def set_initial_state(self, VECTR_INIT_STATE: np.array) -> None:
        '''
        Verify dimension of @VECTR_INIT_STATE.
        Set initial state.

        :param @VECTR_INIT_STATE:               @VECTR_INIT_STATE
        '''

        if not (VECTR_INIT_STATE.size == self._ROWS_STATE_VECTR) or (1 not in VECTR_INIT_STATE.shape):
            raise InvalidVectorDimensions(f"State Vector must have dimension ({self._ROWS_STATE_VECTR} x 1); \
                Current dimensions : ({VECTR_INIT_STATE.shape[0]}x{VECTR_INIT_STATE.shape[1]})")

        self._VECTR_INIT_STATE = np.copy(VECTR_INIT_STATE)


    def set_initial_covariance(self, MAT_INIT_COVAR: np.matrix) -> None:
        pass


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

class InvalidVectorDimensions(Exception):
    pass