import zipfile
import os


def get_meta_data(dataset_path):
    list_dir_dataset_path = os.listdir(dataset_path)
    for file in list_dir_dataset_path:
        if file.endswith('.zip'):
            file_path = os.path.join(dataset_path, file)
            folder_name = os.path.splitext(file)[0]
            extract_path = os.path.join(dataset_path, folder_name)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted {file} successfully.")


def append_list(dataset_path):
    folder_list = []
    audio_files = []
    audio_paths = []
    transcript_paths = []
    for i in os.listdir(dataset_path): #TODO wywalić fora w forze
        if i.endswith('_P'):
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path, i)):
                if 'wav' in j:
                    audio_files.append(j)
                    audio_paths.append(os.path.join(dataset_path, i, j))
                if 'TRANSCRIPT' in j:
                    if 'lock' in j or '._' in j:
                        pass
                    else:
                        transcript_paths.append(os.path.join(dataset_path, i, j))

    return folder_list, audio_paths, transcript_paths

