import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file import FileUtil
from utils.audio import AudioUtils

clear_session()
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Using device: {device}")

if device == "GPU":

    try:
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Memory growth set for GPUs.")
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

target_sr = 16000
batch_size = 8 

audio_dataset = AudioUtils.process_audio_files(FileUtil.DATASET_AUDIO_PATH, target_sr, batch_size)

audio_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same'), 
])

audio_embeddings = []
for audio_batch in audio_dataset:
    audio_batch = tf.expand_dims(audio_batch, axis=-1)
    audio_embedding_batch = audio_model(audio_batch)
    audio_embeddings.append(audio_embedding_batch)

audio_embeddings = np.concatenate(audio_embeddings, axis=0)
print("Shape of audio embeddings:", audio_embeddings.shape)

embedding_audio_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "audio_embeddings.npy")
np.save(embedding_audio_path, audio_embeddings)
print(f"Audio embeddings saved in {embedding_audio_path}")

model_destination_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "audio_embedding.h5")
audio_model.save(model_destination_path)
print(f"Audio embeddings model saved in {model_destination_path}")

