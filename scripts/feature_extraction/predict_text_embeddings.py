import sys
from gensim.models import FastText
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file import FileUtil

MAX_LEN = 80

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def load_text_files(text, model):
    words = preprocess_text(text).split()
    word_embeddings = [model.wv[word] for word in words if word in model.wv]

    if word_embeddings:
        embeddings = np.vstack(word_embeddings)
        if embeddings.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - embeddings.shape[0]
            embeddings = np.pad(embeddings, ((0, pad_width), (0, 0)), mode='constant')
    else:
        embeddings = np.zeros((MAX_LEN, model.vector_size))

    return np.array([embeddings])


def get_text_embeddings(text: str):
    fasttext_model_path = os.path.join(FileUtil.MODEL_FOLDER_PATH, "fasttext_model.model")
    fasttext_model = FastText.load(fasttext_model_path)

    embeddings = load_text_files(text, fasttext_model)

    return embeddings

