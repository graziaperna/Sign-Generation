from tensorflow.keras.models import load_model
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import AudioUtils
from utils.file import FileUtil

def getAudioEmbedding(file_path):
    tf.keras.backend.clear_session()

    target_sr = 16000

    audio_tensor = AudioUtils.load_and_preprocess_audio(tf.convert_to_tensor(file_path), target_sr)
    audio_tensor = tf.expand_dims(audio_tensor, axis=0)  

    audio_model_path = os.path.join(FileUtil.MODEL_FOLDER_PATH, "audio_embedding.h5")
    print(f"Loading model from {audio_model_path}")

    audio_model = load_model(audio_model_path)

    audio_embedding = audio_model.predict(audio_tensor)

    del audio_model
    tf.keras.backend.clear_session()

    return audio_embedding
