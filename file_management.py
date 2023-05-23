import zipfile
import os
import glob
import audio_manipulations
import feature_extraction

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
    for file in glob.glob(dataset_path + '\\*_P\\*'):
        filename = file.split('\\')[-1]
        if 'TRANSCRIPT.csv' in filename:
            audio_manipulations.transcript_file_processing(file)
        if 'wav' in filename:
            feature = feature_extraction.get_features(file)
            features.append(feature)
