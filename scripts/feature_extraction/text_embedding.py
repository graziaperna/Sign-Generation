import numpy as np
import pandas as pd
import re
import os
from gensim.models import FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file import FileUtil

def save_preprocessed_data(file_path):
    '''
      It concatenates all the strings in the 'cleaned_text' column and saves them in a text file.
    '''
    cleaned_text_list = df['cleaned_text'].tolist()

    preprocess_file_path = os.path.join(file_path, "preprocessed_text.txt")

    with open(preprocess_file_path, 'w') as file:
        for line in cleaned_text_list:
            file.write(line + '\n')

    print(f"Preprocessed data have been saved into {preprocess_file_path}.")

def preprocess_text(text):
    '''
      It removes special characters, digits, and converts the text to lowercase.
      '''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def load_excel_files(folder_path):
    '''
      It loads all the text files in the folder and returns a DataFrame.
      '''
    dataframes = []
    files = [filename for filename in os.listdir(folder_path) if filename.endswith('.txt')]
    
    files.sort(key=FileUtil.extract_date)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath_or_buffer=file_path, sep = " -- ", header=None, engine = "python")
            if not df.empty:
                dataframes.append(df)
            else:
                print(f"Il file {filename} è vuoto.")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame() 

def get_embedding_matrix(tokenizer, fasttext_model):
    '''
      It creates an embedding matrix for the tokenizer
    '''
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, fasttext_model.vector_size))
    for word, i in tokenizer.word_index.items():
        if word in fasttext_model.wv:
            embedding_matrix[i] = fasttext_model.wv[word]
    return embedding_matrix

df = load_excel_files(FileUtil.DATASET_TEXT_PATH)

df.columns = ['start_time', 'end_time', 'text']

df['cleaned_text'] = df['text'].apply(preprocess_text)

save_preprocessed_data(FileUtil.GENERATED_FILE_FOLDER_PATH)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

print(df.columns)

max_length = 80
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
X = pad_sequences(sequences, maxlen=max_length)

fasttext_model = FastText(sentences=df['cleaned_text'].apply(lambda x: x.split()), vector_size=100, window=3, min_count=1, sg=1)

embedding_matrix = get_embedding_matrix(tokenizer, fasttext_model)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

input_layer = Input(shape=(X.shape[1],))
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            trainable=True)(input_layer)

print("Embedding layer shape:", embedding_layer.shape)
model = Model(inputs=input_layer, outputs=embedding_layer)

model_destination_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "text_embedding.h5")
model.save(model_destination_path)