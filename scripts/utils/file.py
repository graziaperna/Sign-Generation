import os

class FileUtil:

    GENERATED_FILE_FOLDER_PATH =  os.path.join(os.getcwd(), "generated_file")
    DATASET_PATH = os.path.join(os.getcwd(), "datasets")
    DATASET_TEXT_PATH = os.path.join(DATASET_PATH, "transcripts", "transcripts")
    DATASET_AUDIO_PATH = os.path.join(DATASET_PATH, "audio", "audio")
    DATASER_INTEGRATION_PATH = os.path.join(DATASET_PATH, "integration")
    EMBEDDINGS_FOLDER_PATH = os.path.join(GENERATED_FILE_FOLDER_PATH, "embeddings")
    MODEL_FOLDER_PATH = os.path.join(GENERATED_FILE_FOLDER_PATH, "models")



    def get_subdirectory_file_path(cwd: str, folder_destination:str, file_name:str) -> int:
        folder_path = os.path.join(cwd, folder_destination)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return os.path.join(folder_path, file_name)