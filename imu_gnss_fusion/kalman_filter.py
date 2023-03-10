import numpy as np


class KalmanFilter():
    '''
    Linear Kalman Filter Implementation.
    @Notes: 
    '''
    #
    # ~~~~~             KALMAN FILTER - CONSTRUCTORS              ~~~~~
    #     
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
    
    @classmethod
    def from_matrix_dimensions(cls, 
                                ROWS_STATE_VECTR: int,
                                ROWS_CNTRL_VECTR: int,
                                ROWS_MEAS_VECTR: int, 
                                MAT_PROCESS_NOISE: np.matrix,
                                MAT_MEAS_UNCERT: np.matrix,
                                VECTR_INIT_STATE: np.matrix,
                                MAT_INIT_COVAR: np.matrix
                                ):
        '''
        Alt constrcutor for KalmanFilter. Primarily for inherit compatibility with ExtendedKalmanFilter.
        '''
        return cls(MAT_STATE_TRANSTN=np.zeros((ROWS_STATE_VECTR, ROWS_STATE_VECTR)),
                    MAT_CNTRL=np.zeros((ROWS_STATE_VECTR, ROWS_CNTRL_VECTR)),
                    MAT_OBSRVN=np.zeros((ROWS_MEAS_VECTR, ROWS_STATE_VECTR)),
                    MAT_PROCESS_NOISE=MAT_PROCESS_NOISE,
                    MAT_MEAS_UNCERT=MAT_MEAS_UNCERT,
                    VECTR_INIT_STATE=VECTR_INIT_STATE,
                    MAT_INIT_COVAR=MAT_INIT_COVAR)


    #
    # ~~~~~             KALMAN FILTER - PREDICT              ~~~~~
    #     
    def kalman_predict(self, VECTR_CNTRL: np.array)-> None:
        '''
        Predict the next state estimate based on current state estimate and 
        current control inputs

        :param @VECTR_CNTRL:                 vector containing control inputs
        '''
        self.is_valid_cntrl_vectr(VECTR_CNTRL)
        self.extrapolate_state(VECTR_CNTRL)
        self.extrapolate_covar()


    def is_valid_cntrl_vectr(self, VECTR_CNTRL: np.array)->None:
        '''
        Raise exception if control vector is dimensionally invalid

        :param @VECTR_CNTRL:                 @VECTR_CNTRL
        '''
        if(not VECTR_CNTRL.size == self._ROWS_CNTRL_VECTR) and (1 not in VECTR_CNTRL.shape) and \
            (len(VECTR_CNTRL.shape) != 1):
            raise(InvalidVectorDimensions(f"Control Input Vector must have dimension ({self._ROWS_CNTRL_VECTR} x 1); \
                Current dimensions : ({VECTR_CNTRL.shape[0]}x{VECTR_CNTRL.shape[1]})"))
        pass


    def extrapolate_state(self, VECTR_CNTRL:np.array) -> None:
        '''
        Extrapolate state estimate

        :param @VECTR_CNTRL:                 @VECTR_CNTRL
        '''
        self.VECTR_STATE = np.matmul(self._MAT_STATE_TRANSTN, self.VECTR_STATE) + \
                    np.matmul(self._MAT_CNTRL, VECTR_CNTRL)


    def extrapolate_covar(self) -> None:
        '''
        Extrapolate covariance matrix
        '''
        self.MAT_COVAR = np.matmul( 
                            np.matmul(
                                self._MAT_STATE_TRANSTN,
                                self.MAT_COVAR
                            ), self._MAT_STATE_TRANSTN.T + \
                                self._MAT_PROCESS_NOSE
        )


    #
    # ~~~~~             KALMAN FILTER - UPDATE              ~~~~~
    # 
    def kalman_update(self, 
                    VECTR_MEAS: np.array,
                    **kwargs) -> None:
        '''
        Combine measurements (VECTR_MEAS) with last state estimate 
        (VECTR_STATE) to improve estimate.

        :param @VECTR_MEAS:                 vector containing measurements for state estimate update
        :**(kwargs) @MAT_MEAS_UNCERT:       provide updated measurement uncertainty if applicable
        '''
        self.is_valid_meas_vectr(VECTR_MEAS)

        if('MAT_MEAS_UNCERT' in kwargs.keys()):
            self.set_measurement_uncertainty(kwargs['MAT_MEAS_UNCERT'])

        self.compute_kalman_gain()
        self.update_state_estimate(VECTR_MEAS)
        self.update_covar()


    def is_valid_meas_vectr(self, VECTR_MEAS: np.array) -> None:
        '''
        Raise exception is measurement vector  is dimensionally invalid
        '''
        if(not VECTR_MEAS.size == self._ROWS_MEAS_VECTR) and (1 not in VECTR_MEAS.shape) and \
            (len(VECTR_MEAS.size) == 1):
            raise(InvalidVectorDimensions(f"Measurement Vector must have dimension ({self._ROWS_CNTRL_VECTR} x 1); \
                Current dimensions : ({VECTR_MEAS.shape[0]}x{VECTR_MEAS.shape[1]})"))
        pass

    
    def compute_kalman_gain(self):
        '''
        Compute the Kalman Gain
        '''
        # P_(nxn) * H_(kxn)^T
        _PH_T = np.matmul(self.MAT_COVAR, self._MAT_OBSRVN.T)
        # H_(kxn) * P_(nxn) * H_(kxn)^T
        _HPH_T = np.matmul(self._MAT_OBSRVN, _PH_T)
        self._KALMAN_GAIN = np.matmul(_PH_T,
                                    np.linalg.pinv(
                                        _HPH_T + \
                                        self._MAT_MEAS_UNCERT
                                    ))

    
    def update_state_estimate(self, VECTR_MEAS):
        '''
        Update the state estimate using measurement and 
        most recent kalman gain

        :param @VECTR_MEAS:                 @VECTR_MEAS
        '''
        self.VECTR_STATE = self.VECTR_STATE +  \
                        np.matmul(self._KALMAN_GAIN,
                                    (VECTR_MEAS - \
                                    np.matmul(self._MAT_OBSRVN,
                                            self.VECTR_STATE)
                                )                    
                        )
        

    def update_covar(self):
        '''
        Update the covariance matrix after state update
        '''
        # I_(nxn) * K_(nxk)^T
        _I_KH = np.eye(self._ROWS_STATE_VECTR) -\
                    np.matmul(self._KALMAN_GAIN , self._MAT_OBSRVN)
        self.MAT_COVAR = np.matmul( np.matmul(_I_KH,
                                             self.MAT_COVAR),
                                    _I_KH.T) + \
                        np.matmul(np.matmul(self._KALMAN_GAIN,
                                            self._MAT_MEAS_UNCERT),
                                    self._KALMAN_GAIN.T)
    

    #
    # ~~~~~         KALMAN FILTER - SETTERS & GETTERS       ~~~~~
    # 

    # Setter Method
    def set_state_transition_mat(self, MAT_STATE_TRANSTN : np.matrix) -> None:
        '''
        Set the state transition matrix for the Kalman Filter.

        This is done only once for Linear Kalman Filter, in EFK usecase, this will 
        be called upon every lienarization (i.e each step).

        Validate dimensions, and allow alternative contructor pathways 

        :param @MAT_STATE_TRANSTN:          @MAT_STATE_TRANSTN
        '''
        _n = MAT_STATE_TRANSTN.shape[0]
        _x = MAT_STATE_TRANSTN.shape[1]

        if(not _n == _x):
            raise InvalidMatrixDimensions(f"State Transtion must be a square matrix. Current dimensions: ({_n} x {_x})")

        # Alt constructor support - init with ROWS_STATE_VECTOR
        if hasattr(self, '_ROWS_STATE_VECTR'):

            if (not _n == self._ROWS_STATE_VECTR):

                raise InvalidMatrixDimensions(f"State Matrix dimensions ({_n} x {_x}), invalid. \
                Must have dimensions ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR})")
            
            # Alt constructor for use with EKF 
            # Setter will be called alot, therefore remove "np.copy"
            self._MAT_STATE = MAT_STATE_TRANSTN

        # Default constructor path
        else:
            self._MAT_STATE_TRANSTN = np.copy(MAT_STATE_TRANSTN) 
            self._ROWS_STATE_VECTR = _n
           
        

    # Setter Method 
    def set_control_mat(self, MAT_CNTRL: np.matrix) -> None:
        '''
        Set the control matrix for KF/EKF.

        Validate dimensions, and allow alternative contructor pathways 

        :param @MAT_CNTRL:                  @MAT_CNTRL
        '''
        _n = MAT_CNTRL.shape[0]
        _m = MAT_CNTRL.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Control matrix dimensions ({_n} x {_m}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Control matrix must have the dimensions ({self._ROWS_STATE_VECTR } x m)")

        if hasattr(self, '_ROWS_CNTRL_VECTR'):
            
            if (not _m == self._ROWS_CNTRL_VECTR):
                raise InvalidMatrixDimensions(f"Control Matrix dimensions ({_n} x {_m}), invalid. \
                    Must have dimensions ({self._ROWS_STATE_VECTR} x {self._ROWS_CNTRL_VECTR})")
                
            self._MAT_CNTRL = MAT_CNTRL

        else:
            self._MAT_CNTRL = np.copy(MAT_CNTRL)
            self._ROWS_CNTRL_VECTR = _m


    # Setter Method
    def set_observation_mat(self, MAT_OBSRVN: np.matrix) -> None:
        '''
        Set the observation matrix for KF/EKF.

        Validate dimensions, and allow alternative contructor pathways 

        :param @MAT_OBSRVN:                 @MAT_OBSRVN
        '''
        _k = MAT_OBSRVN.shape[0]
        _n = MAT_OBSRVN.shape[1]

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Observation Matrix dimensions ({_k} x {_n}), inconsistent \
                 with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                    Observation Matrix must have the dimensions (k x {self._ROWS_STATE_VECTR })")


        if hasattr(self, '_ROWS_MEAS_VECTR'):

            if not (_k == self._ROWS_MEAS_VECTR):
                raise InvalidMatrixDimensions(f"Observation Matrix dimensions ({_k} x {_n}), invalid. \
                    Must have dimensions ({self._ROWS_MEAS_VECTR} x {self._ROWS_STATE_VECTR})")

            self._MAT_OBSRVN = MAT_OBSRVN


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

        if(not _k == self._ROWS_MEAS_VECTR):
            raise InconsistentMatrixDimensions(f"Measurement Uncertainty Matrix dimensions ({_k} x {_x}), \inconsistent \
                with observation matrix ({self._ROWS_MEAS_VECTR} x {self._ROWS_MEAS_VECTR}).\n \
                Measurement Uncertainty Matrix must have the dimensions ({self._ROWS_MEAS_VECTR}  x {self._ROWS_MEAS_VECTR})")

        if(not _k == _x):
            raise InvalidMatrixDimensions(f"Measurement Uncertainty must be a square matrix. \
             Current dimensions: ({_k} x {_x})")
            
        self._MAT_MEAS_UNCERT = MAT_MEAS_UNCERT


    # Setter Method
    def set_initial_state(self, VECTR_INIT_STATE: np.array) -> None:
        '''
        Verify dimension of @VECTR_INIT_STATE provided by user.
        Set initial state.

        :param @VECTR_INIT_STATE:               @VECTR_INIT_STATE
        '''

        if (VECTR_INIT_STATE.size != self._ROWS_STATE_VECTR) and (1 not in VECTR_INIT_STATE.shape) \
            and (len(VECTR_INIT_STATE.size) != 1):
            raise InvalidVectorDimensions(f"State Vector must have dimension ({self._ROWS_STATE_VECTR} x 1); \
                Current dimensions : ({VECTR_INIT_STATE.shape})")

        self.VECTR_STATE = np.copy(VECTR_INIT_STATE)


    # Setter Method
    def set_initial_covariance(self, MAT_INIT_COVAR: np.matrix) -> None:
        '''
        Verify dimension of @MAT_INIT_COVAR wrt to @MAT_STATE_TRANSITION
        Set initial COVARIANCE Matrix

        :param @MAT_INIT_COVAR:                 @MAT_INIT_COVAR
        '''
        _n = MAT_INIT_COVAR.shape[0]
        _x = MAT_INIT_COVAR.shape[1]

        if (not _n == _x):
            raise InvalidMatrixDimensions(f"Covariance must be a square matrix.\
                 Current dimensions: ({_n} x {_x})")

        if(not _n == self._ROWS_STATE_VECTR):
            raise InconsistentMatrixDimensions(f"Covariance Matrix dimensions ({_n} x {_x}), inconsistent \
                with state transition matrix ({self._ROWS_STATE_VECTR} x {self._ROWS_STATE_VECTR}).\n \
                Covariance Matrix must have the dimensions ({self._ROWS_STATE_VECTR}  x {self._ROWS_STATE_VECTR})")
            
        self.MAT_COVAR = np.copy(MAT_INIT_COVAR)

    



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