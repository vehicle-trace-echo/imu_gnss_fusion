import numpy as np
from imu_gnss_fusion.kalman_filter import KalmanFilter


class ExtendedKalmanFilter(KalmanFilter):
    '''
    Extended Kalman Filter class
    @Notes:
    Roughly uses the KalmanFilter class as the base class.
    '''

    def __init__(self,
                state_trnstn_mdl,
                state_trnstn_jcbn_state,
                state_trnstn_jcbn_cntrl,                
                measmnt_mdl,
                measmnt_mdl_jcbn_state,
                MAT_PROCESS_NOISE: int,
                MAT_MEAS_UNCERT: int,
                VECTR_INIT_STATE: int,
                MAT_INIT_COVAR: int,
                ROWS_VECTR_STATE: int,
                ROWS_CNTRL_VECTR: int,
                ROWS_MEAS_VECTR: int,
                ):     
        '''    
        :param @state_trnstn_mdl:           (@function) 
                                            user-implemented function which evaluates state transition 
                                            must be implemented with the following input and output parameters
                                            <inputs: current_state, current_inputs>
                                            <output: predicted_state>

        :param @state_trnstn_jcbn_state:    (@function) 
                                            user-implemented fucntion to evaluate the jacobian matrix
                                            with respect to STATE VECTOR at current state and inputs.
                                            <inputs: current_state, predicted_state, current_inputs>
                                            <outputs: jacobian_matrix>
        
        :param @state_trnstn_jcbn_cntrl:    (@function) 
                                            user-implemented fucntion to evaluate the jacobian matrix
                                            with respect to CONTROL VECTOR at current state and inputs
                                            <inputs: current_state, predicted_state, current_inputs>
                                            <output: jacobian_matrix> 
        
        :param @measmnt_mdl:                (@function)
                                            user-implemented function which connects states to measurements
                                            <inputs: current_state> 
                                            <outputs: ?>

        :param @measmnt_mdl_jcbn_state:     (@function)
                                            user-implemented function to evaluate the jacobian matrix of measurement
                                            model with respect to state.
                                            <inputs: current_state, predicted_state>
                                            <outputs: jacobian_matrix (EKF equiv. of Observation Matrix (H))

        :param @MAT_PROCESS_NOISE:          process noise; uncertainty in control inputs and process model
        :param @MAT_MEAS_UNCERT:            measurement noise; uncertainty in observations 
        :param @VECTR_INIT_STATE:           initial state vector, guessed or provided by a different process
        :param @MAT_INIT_COVAR:             initial covariance matrix; uncertainty in initial state vector
        '''
        super().from_matrix_dimensions(ROWS_VECTR_STATE,
                                    ROWS_CNTRL_VECTR,
                                    ROWS_MEAS_VECTR,
                                    MAT_PROCESS_NOISE,
                                    MAT_MEAS_UNCERT,
                                    VECTR_INIT_STATE,
                                    MAT_INIT_COVAR)

        self.set_initial_state(VECTR_INIT_STATE)
        self.set_initial_covariance(MAT_INIT_COVAR)
        self.set_process_noise(MAT_PROCESS_NOISE)
        self.set_measurement_uncertainty(MAT_MEAS_UNCERT)

        self._state_trnstn_mdl = state_trnstn_mdl
        self._get_state_trnstn_jcbn_state = state_trnstn_jcbn_state
        self._get_state_trnstn_jcbn_cntrl = state_trnstn_jcbn_cntrl
        self._measmnt_mdl = measmnt_mdl
        self._get_measmnt_mdl_jcbn_state = measmnt_mdl_jcbn_state


    def kalman_predict(self, VECTR_CNTRL: np.array) -> None:
        '''
        **Overwrites the kalman_predict implementation from KalmanFilter class

        Predict the next state estimate based on current state estimate and 
        current control inputs

        :param @VECTR_CNTRL:                 vector containing control inputs
        '''
        self.extrapolate_state()

        self.set_state_transition_mat(
                        self._get_state_strnstn_jcnb_state(
                                    self.VECTR_STATE, self._VECTR_STATE_NEXT, VECTR_CNTRL
                                                          )
                                    )

        self.set_control_mat(
                    self._get_state_trnstn_jcbn_cntrl(
                                self.VECTR_STATE, self._VECTR_STATE_NEXT, VECTR_CNTRL
                                )
                            )
        self.extrapolate_covar()
        self.VECTR_STATE = self._VECTR_STATE_NEXT



    def extrpolate_state(self, VECTR_CNTRL:np.array) -> None:
        '''
        **Overwrites the extrapolate_state implementation from KalmanFilter class
        Extrapolate state estimate

        :param @VECTR_CNTRL:                vector containing control inputs                 
        '''
        self._VECTR_STATE_NEXT = self._state_trnstn_mdl(self.VECTR_STATE,  VECTR_CNTRL)
        
    
    def kalman_update(self,
                     VECTR_MEAS: np.array,
                     **kwargs) -> None:
        '''
        **Overwrites the kalman_update implementation from KalmanFilter class

        Combine measurements (VECTR_MEAS) with last state estimate 
        (VECTR_STATE) to improve estimate.

        :param @VECTR_MEAS:                 vector containing measurements for state estimate update
        :**(kwargs) @MAT_MEAS_UNCERT:       provide updated measurement uncertainty if applicable
        '''
        self.is_valid_meas_vectr(VECTR_MEAS)

        if('MAT_MEAS_UNCERT' in kwargs.key()):
            self.set_measurement_uncertainty(kwargs['MAT_MEAS_UNCERT'])

        self.set_observation_mat(
            self._get_measmnt_mdl_jcbn_state(
                self.VECTR_STATE))
        self.compute_kalman_gain()
        self.update_state_estimate(VECTR_MEAS)
        self.update_covar()


    def update_state_estimate(self, VECTR_MEAS):
        '''
        **Overwrites the update_state_estimate implementation from KalmanFilter class

        Update the state estimate using measurement and 
        most recent kalman gain
        '''
        return self.VECTR_STATE + np.matmul(
                                self._KALMAN_GAIN,
                                (VECTR_MEAS - \
                                self._measmnt_mdl(self.VECTR_state)
                                )
        )


        
