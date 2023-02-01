import math
import numpy as np
from kalman_filter import KalmanFilter


class ExtendedKalmanFilter(KalmanFilter):
    '''
    Extended Kalman Filter class
    @Notes:
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
        self.VECTR_STATE = self._VECTR_STATE_NEXT.copy()



    def extrpolate_state(self, VECTR_CNTRL:np.array) -> None:
        self._VECTR_STATE_NEXT = self._state_trnstn_mdl(self.VECTR_STATE,  VECTR_CNTRL)
        
    
    def kalman_update(self,
                     VECTR_MEAS: np.array,
                     **kwargs) -> None:
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
        return self.VECTR_STATE + np.matmul(
                                self._KALMAN_GAIN,
                                (VECTR_MEAS - \
                                self._measmnt_mdl(self.VECTR_state)
                                )
        )


        
