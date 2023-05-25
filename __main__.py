import file_management
import config


if __name__ == "__main__":
  # file_management.extract_zip_files(config.DATASET_DIR)
  file_management.read_files_daic_woz(config.DAIC_WOZ_DIR)
  file_management.get_EATD_corpus()
