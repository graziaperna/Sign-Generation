# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gz29IzsCS0IZGnCK4r9d6IXJ6k4vofKE
"""

# !pip install keras==2.15.0
# !pip install shap==0.45.1
# !pip install tensorflow==2.15.0
# !pip install tensorflow-estimator==2.15.0
# !pip install tensorflow-io-gcs-filesystem==0.37.0
# !pip install tf_keras==2.16.0

# !pip install tensorflow==2.15.1
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file import FileUtil

text_model_path = os.path.join(FileUtil.MODEL_FOLDER_PATH, "text_embedding.h5")
max_length = 80

text_model = load_model(text_model_path)

dataset_text_path = os.path.join(FileUtil.GENERATED_FILE_FOLDER_PATH, "preprocessed_text.txt")

with open(dataset_text_path, 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file.readlines()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
text_sequences = tokenizer.texts_to_sequences(sentences)

padded_text = pad_sequences(text_sequences, maxlen=max_length)

padded_text = padded_text[..., None]

text_embeddings = text_model.predict(padded_text)
print("Embeddings shape:", text_embeddings.shape)

embeddings_file_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.EMBEDDINGS_FOLDER_PATH, "text_embeddings.txt")

np.savetxt(embeddings_file_path, text_embeddings.reshape(text_embeddings.shape[0], -1), delimiter=' ', fmt='%.6f')