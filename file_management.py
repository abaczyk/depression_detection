import zipfile
import os
import numpy as np
import audio_manipulations
import feature_extraction
import csv
import config

features = []
labels = []

def extract_zip_files(dataset_path):
    list_dir_dataset_path = os.listdir(dataset_path)
    for file in list_dir_dataset_path:
        if file.endswith('.zip'):
            file_path = os.path.join(dataset_path, file)
            folder_name = os.path.splitext(file)[0]
            extract_path = os.path.join(dataset_path, folder_name)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted {file} successfully.")


def read_files_from_dir(dataset_path):
    with open('all_labels.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id = row['ID']
            label = row['PHQ_bin']
            transcript_filename = f'{dataset_path}\\{id}_P\\{id}_TRANSCRIPT.csv'
            audio_filename = f'{dataset_path}\\{id}_P\\{id}_AUDIO.wav'
            if os.path.isfile(transcript_filename):
                audio_manipulations.transcript_file_processing(transcript_filename)
            if os.path.isfile(audio_filename):
                feature = feature_extraction.get_features(audio_filename)
                features.append(feature)
                labels.append(label)    

    csvfile.close()

def save_features(prefix, audio_features):
    if not os.path.exists(os.path.join(prefix, 'AudioFeaturesEATD')):
        os.mkdir(os.path.join(prefix, 'AudioFeaturesEATD'))
    np.savez(os.path.join(prefix, 'AudioFeaturesEATD\extracted_audio_features.npz'), audio_features)
    print('Saved audio features in: {}/AudioFeaturesEATD'.format(config.SAVE_DIR))

def get_EATDcorpus():
    audio_features = []
    audio_file_name = 'EATD-corpus'

    # total files 105
    for index in range(105): #todo usunąć to
        feature_extraction.extract_features(index + 1, audio_features, audio_file_name)

    save_features(config.SAVE_DIR, audio_features)