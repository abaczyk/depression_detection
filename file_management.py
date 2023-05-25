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

def save_features(prefix, audio_features, folder_name):
    if not os.path.exists(os.path.join(prefix, folder_name)):
        os.mkdir(os.path.join(prefix, folder_name))
    np.savez(os.path.join(prefix, '{}\extracted_audio_features.npz'.format(folder_name)), audio_features)
    print('Saved audio features in: {}/{}'.format(config.SAVE_DIR, folder_name))
    
def read_files_daic_woz(dataset_path):
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
                feature = feature_extraction.get_features(*feature_extraction.read_file(audio_filename))
                features.append(feature)
                labels.append(label)    
        save_features(config.SAVE_DIR, features, 'AudioFeaturesDAIC-WOZ')
    csvfile.close()

def get_EATD_corpus():
    audio_features = []
    prefix = (config.EATD_DIR.split('\\'))[0]
    folder_name = (config.EATD_DIR.split('\\'))[1]
    # total files 105
    for index in range(105):
        feature_extraction.extract_features_eatd(index + 1, audio_features, folder_name, prefix)

    save_features(config.SAVE_DIR, audio_features, 'AudioFeaturesEATD')