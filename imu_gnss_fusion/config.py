GRAV_MAGNTD = 9.8066

IMU_CFG = {
    'range' : {
        'imu_max': 32767,
        'imu_min': -32768,
        'units': 'ADC'
             },  

    'accel' : {
        'range': {
            'accel_min': -2,
            'accel_max': 2,
            'units': 'g'
                    },

        'bias' : {
                'x' : -1.05954896e-01,
                'y' : -1.60772046e-01,
                'z' : -2.39030854e-02,
                'units':'METERSPERSECONDS',
                },
        'misalign':
                {
                    'alpha_yz': 2.71122385e-05,
                    'alpha_zy': -4.25645029e-03,
                    'alpha_zx': 2.18810007e-03,
                },
        'scale':
                {
                'x': 9.96700502e-01,
                'y': 9.96589996e-01, 
                'z': 9.87936898e-01,
                }
            },
        
    'gyro' : {
        
        'range': {
                'gyro_min' : -250,
                'gyro_max' : 250,
                'units': 'DEGREESPERSECOND'
            },
                        
        'bias' :
            {
            'x': -0.6584513260335345,
            'y': 1.202140150793541, 
            'z':-0.1227491118389093,
            'units': 'DEGREESPERSECONDS'
            },
        'misalign':
                {
            'gamma_yz': 0,
            'gamma_zy': 0,
            'gamma_zx': 0,
                },
        'scale':
                {
                'x': 1,
                'y': 1, 
                'z': 1,
                }                
            },
}
