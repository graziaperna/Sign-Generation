from tensorflow.keras.models import load_model
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import AudioUtils
from utils.file import FileUtil

clear_session()  
tf.config.set_visible_devices([], 'GPU')

target_sr = 16000
batch_size = 8

audio_dataset = AudioUtils.process_audio_files(FileUtil.DATASET_AUDIO_PATH, target_sr, batch_size)

audio_model_path = os.path.join(FileUtil.MODEL_FOLDER_PATH, "audio_embedding.h5")
print(f"Loading model from {audio_model_path}")

audio_model = load_model(audio_model_path)
audio_model.compile(optimizer='adam', loss='mse')

audio_embeddings = []

for batch_data in audio_dataset:
    embeddings_batch = audio_model.predict(batch_data)
    audio_embeddings.append(embeddings_batch)
    
    del batch_data
    tf.keras.backend.clear_session()

audio_embeddings = np.concatenate(audio_embeddings, axis=0)
embeddings_file_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.EMBEDDINGS_FOLDER_PATH, "audio_embeddings.txt")

np.savetxt(embeddings_file_path, audio_embeddings.reshape(audio_embeddings.shape[0], -1), delimiter=' ', fmt='%.6f')

del audio_model
clear_session()
