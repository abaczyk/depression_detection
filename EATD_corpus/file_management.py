
import sys
import os
import extract_audio_features
import numpy as np



def save_features(prefix, audio_features):
    if not os.path.exists(os.path.join(prefix, 'AudioFeaturesEATD')):
        os.mkdir(os.path.join(prefix, 'AudioFeaturesEATD'))
    np.savez(os.path.join(prefix, 'AudioFeaturesEATD\extracted_audio_features.npz'), audio_features)
    print('Saved audio features in: {}/AudioFeaturesEATD'.format(prefix))

def get_EATDcorpus():
    if os.path.exists(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".")), 'EATD-corpus')):
        prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
    if os.path.exists(os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'EATD-corpus')):
        prefix = os.path.abspath(os.path.join(os.getcwd(), ".."))
    else:
        print('Brak pliku z baza')
        return


    audio_features = []
    audio_file_name = 'EATD-corpus'

    # total files 105
    for index in range(105):
        extract_audio_features.extract_features(index + 1, audio_features, audio_file_name, prefix)

    save_features(prefix, audio_features)


