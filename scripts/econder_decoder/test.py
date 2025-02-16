from tensorflow.keras.models import load_model
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf

from feature_extraction import predict_text_embeddings
from feature_extraction import predict_audio_embeddings

print(tf.__version__)
current_dir = os.path.dirname(os.path.abspath(__file__))

generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
xlsx_file_folder_path = os.path.join(current_dir, "../../datasets/transcripts/transcripts/Tg_Noi_Lis_01_06_2022.txt")
audio_file_folder_path = os.path.join(current_dir, "../../datasets/audio/Tg_Noi_Lis_01_06_2022.mp3")

gen_path = os.path.join(generated_file_folder_path, "models/generator_model.keras")
audio_embedding_path = os.path.join(generated_file_folder_path, "embeddings/audio_sample.mp3")

import pandas as pd

new_text = "Ciao, mi chiamo Grazia e sto facendo un test per vedere quanto questo progetto faccia schifo. Per favore Dio aiutami"
df = pd.read_csv(xlsx_file_folder_path, sep=" -- ", header=None, encoding="latin1", engine="python")
df.columns = ['start_time', 'end_time', 'text']
print(df["text"])
text_embeddings = predict_text_embeddings.get_text_embeddings(" ".join(df["text"].astype(str)))
print("Generated embedding for input text.")

audio_embeddings = predict_audio_embeddings.getAudioEmbedding(audio_file_folder_path)
print("Generated embedding for input audio.")


print(f"Initial text_embeddings shape: {text_embeddings.shape}") #(100,)
print(f"Initial audio_embeddings shape: {audio_embeddings.shape}")

pca_path = os.path.join(generated_file_folder_path, "models/pca_model.pkl")

text_embeddings = np.expand_dims(text_embeddings, axis=(0, 1))
audio_embeddings = np.expand_dims(audio_embeddings, axis=1) 


print(f"new text shape: {text_embeddings.shape}")
print(f"audio_embeddings shape: {audio_embeddings.shape}")
model = tf.keras.models.load_model(gen_path)
model.summary()
generated_betas, generated_rot_mats = model.predict([text_embeddings, audio_embeddings])

output_dir = os.path.join(generated_file_folder_path, "models/embeddings/")
os.makedirs(output_dir, exist_ok=True)
print("Generated betas:", generated_betas)
print("Generated rotation matrices:", generated_rot_mats)
np.save(os.path.join(output_dir, "generated_betas.npy"), generated_betas)
np.save(os.path.join(output_dir, "generated_rot_mats.npy"), generated_rot_mats)



