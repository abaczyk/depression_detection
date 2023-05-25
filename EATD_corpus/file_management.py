
import sys
import os
import extract_audio_features
import numpy as np
import csv



def save_features(prefix, audio_features):
    if not os.path.exists(os.path.join(prefix, 'AudioFeaturesEATD')):
        os.mkdir(os.path.join(prefix, 'AudioFeaturesEATD'))
    np.savez(os.path.join(prefix, 'AudioFeaturesEATD\extracted_audio_features.npz'), audio_features)
    print('Saved audio features in: {}/AudioFeaturesEATD'.format(prefix))


def get_EATDcorpus():
    EATD_corpus_folder = 'EATD-corpus'

    if os.path.exists(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".")), EATD_corpus_folder)):
        prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
    elif os.path.exists(os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), EATD_corpus_folder)):
        prefix = os.path.abspath(os.path.join(os.getcwd(), ".."))
    else:
        print('Missing database folder')
        return


    audio_features = []
    labels = []
    EATD_corpus_path = os.path.join(prefix, EATD_corpus_folder)

    with open(os.path.join(prefix, 'labels.csv'), 'w', newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)

        for folder_name in os.listdir(EATD_corpus_path):
            extract_audio_features.extract_features(folder_name, audio_features, labels, EATD_corpus_path)

        for item in labels:
            csv_writer.writerow(item)

    save_features(prefix, audio_features)


