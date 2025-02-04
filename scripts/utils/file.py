import os
import re

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
    

    def extract_date(filepath):
        """
       It extracts the date from the file name in the format DD_MM_YYYY.
        """
        filename = os.path.basename(filepath)
        if "preprocessed" not in filename and "interpolated" not in filename:
            match = re.search(r'(\d{2})_(\d{2})_(\d{4})', filename)
            if match:
                day, month, year = map(int, match.groups())
                return year, month, day
            else:
                raise ValueError(f"Format filename not valid: {filename}")
            

    def extract_date_and_number_from_filename(filename):
        """
        It extracts day, mounth, year and number from the format 'Tg_Noi_Lis_dd_mm_yyyy_n.pkl'.
        """
        match = re.match(r"Tg_Noi_Lis_(\d{2})_(\d{2})_(\d{4})_(\d+)\.pkl", filename)
        if match:
            day, month, year, number = match.groups()
            return (int(year), int(month), int(day), int(number))
        else:
            raise ValueError(f"Format filename not valid: {filename}")
