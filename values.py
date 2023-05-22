import numpy as np

EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
                      'FREQ_BINS': 40,
                      'DATASET_IS_BACKGROUND': False,
                      'WHOLE_TRAIN': False,
                      'WINDOW_SIZE': 1024,
                      'OVERLAP': 50,
                      'SNV': True,
                      'SAMPLE_RATE': 16000,
                      'REMOVE_BACKGROUND': True}

windows_function = np.hanning(EXPERIMENT_DETAILS['WINDOW_SIZE'])
fmin = 0
fmax = EXPERIMENT_DETAILS['SAMPLE_RATE'] / 2
hop_size = EXPERIMENT_DETAILS['WINDOW_SIZE'] -\
           round(EXPERIMENT_DETAILS['WINDOW_SIZE'] * (EXPERIMENT_DETAILS['OVERLAP'] / 100))


if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', EXPERIMENT_DETAILS['FEATURE_EXP']]

zip_directory = ''