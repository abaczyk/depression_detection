import zipfile
import os
import glob
import audio_manipulations
import feature_extraction
import csv

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
