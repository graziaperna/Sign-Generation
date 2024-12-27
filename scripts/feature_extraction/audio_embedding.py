import os
import librosa
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

model = tf.keras.Sequential([
    tf.keras.Input(shape=(target_sr, 1)),
    tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
])

for audio_batch in audio_dataset:
    embeddings = model(audio_batch)

model_destination_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "audio_embedding.h5")
model.save(model_destination_path)