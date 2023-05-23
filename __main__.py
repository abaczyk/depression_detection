import unzip_files
import config

if __name__ == "__main__":
  unzip_files.get_meta_data(config.DATASET_DIR)
  unzip_files.append_list(config.DATASET_DIR)
