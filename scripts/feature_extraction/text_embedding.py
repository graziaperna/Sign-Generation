import os
import re
import pandas as pd
import numpy as np
from gensim.models import FastText
import sys
import tensorflow as tf

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

fasttext_model = FastText(sentences=df['cleaned_text'].apply(lambda x: x.split()), vector_size=128, window=3, min_count=1, sg=1)

max_len = max(len(text.split()) for text in df['cleaned_text'])
print("max_len: ", max_len)

text_embeddings = []
for text in df['cleaned_text']:
    word_embeddings = [fasttext_model.wv[word] for word in text.split() if word in fasttext_model.wv]
    if word_embeddings:
        padded_embeddings = np.vstack(word_embeddings)
        if padded_embeddings.shape[0] < max_len:
            pad_width = max_len - padded_embeddings.shape[0]
            padded_embeddings = np.pad(padded_embeddings, ((0, pad_width), (0, 0)), mode='constant')
    else:
        padded_embedding = np.zeros((max_len, 1))
    text_embeddings.append(padded_embeddings)

text_embeddings = np.array(text_embeddings)


print("Shape of text embeddings:", text_embeddings.shape)

embedding_text_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "text_embeddings.npy")
np.save(embedding_text_path, text_embeddings)
print(f"Text embeddings saved in {embedding_text_path}")

fasttext_model_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "fasttext_model.model")
fasttext_model.save(fasttext_model_path)
print(f"FastText model saved at {fasttext_model_path}")