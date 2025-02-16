import numpy as np
import pandas as pd
import re
import os
import pickle
from gensim.models import FastText
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file import FileUtil

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def load_text_files(folder_path):
    dataframes = []
    files = [filename for filename in os.listdir(folder_path) if filename.endswith('.txt')]
    files.sort(key=FileUtil.extract_date)
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep=" -- ", header=None, engine="python")
        if not df.empty:
            dataframes.append(df)
        else:
            print(f"File {filename} is empty.")
    
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

df = load_text_files(FileUtil.DATASET_TEXT_PATH)
df.columns = ['start_time', 'end_time', 'text']
df['cleaned_text'] = df['text'].apply(preprocess_text)

fasttext_model = FastText(
    sentences=df['cleaned_text'].apply(lambda x: x.split()), 
    vector_size=128, 
    window=3, 
    min_count=1, 
    sg=1
)

print("FastText vocabulary words:", list(fasttext_model.wv.key_to_index.keys())[:10])

embedding_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "fasttext_word_embeddings.npy")


fasttext_model_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "fasttext_model.model")
fasttext_model.save(fasttext_model_path)
print(f"FastText model saved at {fasttext_model_path}")


original_text_embeddings = fasttext_model.wv.vectors  # shape (26346, 100)
num_audio_samples = 128

text_groups = np.array_split(original_text_embeddings, num_audio_samples)
aggregated_text_embeddings = np.array([group.mean(axis=0) for group in text_groups])
print("Shape degli embedding testuali aggregati:", aggregated_text_embeddings.shape)

np.save(embedding_path, aggregated_text_embeddings)
print(f"Embedding saved in {embedding_path}")

