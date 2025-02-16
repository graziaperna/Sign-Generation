from io import StringIO
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file import FileUtil

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def get_text_embeddings(text: str):
    fasttext_model_path = os.path.join(FileUtil.MODEL_FOLDER_PATH, "fasttext_model.model")
    fasttext_model = FastText.load(fasttext_model_path)

    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    fasttext_embeddings = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]

    if fasttext_embeddings:
        return np.mean(fasttext_embeddings, axis=0)
    
    return np.zeros(fasttext_model.vector_size)
