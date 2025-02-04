import os
import librosa
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from file import FileUtil


class AudioUtils:

    @staticmethod
    def load_and_preprocess_audio(file_path, target_sr=16000):
        """
        It loads an audio file, converts it to a tensor, and resamples it to target_sr.
        """
        audio, sr = librosa.load(file_path.numpy().decode('utf-8'), sr=target_sr, mono=True)
        
        if len(audio) < target_sr:
            audio = tf.pad(audio, [[0, target_sr - len(audio)]])
        else:
            audio = audio[:target_sr]

        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    
        return audio_tensor

    def process_audio_files(dataset_audio_path, target_sr=16000, batch_size=8):
        """
        It creates a data pipeline to load and preprocess mp3 audio files.
        """
        audio_files = sorted([os.path.join(dataset_audio_path, f) for f in os.listdir(dataset_audio_path) if f.endswith('.mp3')], key = FileUtil.extract_date)
        print(f"Audio file name: {(audio_files)}")
        audio_dataset = tf.data.Dataset.from_tensor_slices(audio_files)
        audio_dataset = audio_dataset.map(lambda x: tf.py_function(AudioUtils.load_and_preprocess_audio, [x, target_sr], tf.float32))
        audio_dataset = audio_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return audio_dataset

