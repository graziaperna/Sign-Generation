import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import h5py
import os
import sys
import pandas as pd
import torch
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from econder_decoder import const
from utils.file import FileUtil

LATENT_DIM = 256

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
audio_model_path = os.path.join(generated_file_folder_path, "models/audio_embedding.h5")
text_model_path = os.path.join(generated_file_folder_path, "models/text_embedding.h5")
movement_model_path = os.path.join(generated_file_folder_path, "models/movement_embedding.h5")

def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    return tensor

def normalize(data, name):
    if data is None or len(data) == 0:
        print(f"Data for {name} is empty or None.")
        return data
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [tensor_to_numpy(item) if isinstance(item, torch.Tensor) else item for item in value]
            else:
                data[key] = tensor_to_numpy(value)
        data = np.concatenate([np.ravel(value) for value in data.values()])
    elif isinstance(data, list):
        data = np.concatenate([np.ravel(tensor_to_numpy(item)) for item in data])
    elif isinstance(data, np.ndarray):
        data = np.ravel(data)
    elif isinstance(data, torch.Tensor):
        data = tensor_to_numpy(data).ravel()
    else:
        raise TypeError("Data type not supported.")
    
    print(f"Normalizing {name}, shape {data.shape}")
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

combined_data_list = []
files = []

for root, dirs, filenames in os.walk(FileUtil.DATASER_INTEGRATION_PATH):
    dirs.sort(key=FileUtil.extract_date)
    filenames = sorted([f for f in filenames if f.endswith('.pkl')], key=FileUtil.extract_date_and_number_from_filename)
    for filename in filenames:
        files.append(os.path.join(root, filename))

for i, file in enumerate(files):
    try:
        data = pd.read_pickle(file)
        print(f"File {file} loading succeeded.")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue
    
    rot_mats = data.get('rot_mats')
    betas = data.get('betas')
    expression = data.get('expression')
    phi = data.get('phi')
    joints = data.get('joints')
    camera = data.get('camera')
    focal = data.get('focal')
    
    if any(x is None or len(x) == 0 for x in [rot_mats, betas, expression, phi, joints, camera, focal]):
        print(f"Skipping file due to missing or empty data: {file}")
        continue
    
    joints_normalized = normalize(joints, 'joints')
    betas_normalized = normalize(betas, 'betas')
    phi_normalized = normalize(phi, 'phi')
    rot_mats_normalized = normalize(rot_mats, 'rot_mats')
    expression_normalized = normalize(expression, 'expression')
    camera_normalized = normalize(camera, 'camera')
    focal_normalized = normalize(focal, 'focal')
    
    combined_data = np.concatenate([
        joints_normalized, betas_normalized, phi_normalized,
        rot_mats_normalized, expression_normalized,
        camera_normalized, focal_normalized
    ], axis=-1)
    
    combined_data_list.append(combined_data)
    
    del data, rot_mats, betas, expression, phi, joints, camera, focal
    gc.collect()
    print(f"File {file} processed and memory freed.")
    print(f"Files remaining: {len(files) - (i + 1)}")

if not combined_data_list:
    print("No data available to concatenate.")
else:
    print("Data concatenation starting...")
    all_combined_data = np.concatenate(combined_data_list, axis=0)
    print("Data concatenation completed.")

if all_combined_data.ndim == 1:
    all_combined_data = np.expand_dims(all_combined_data, axis=1)

movement_emb_size = all_combined_data.shape[1]

with h5py.File(text_model_path, "r") as f:
    embeddings = f[const.TEXT_EMBEDDING_NAME]
    text_emb_size = embeddings.shape[1]

with h5py.File(audio_model_path, "r") as f:
    kernel = f[const.AUDIO_EMBEDDING_NAME]
    audio_emb_size = kernel.shape[1]

# **Encoder**
text_input = layers.Input(shape=(text_emb_size,))
audio_input = layers.Input(shape=(audio_emb_size,))
movement_input = layers.Input(shape=(movement_emb_size,))

merged = layers.Concatenate()([text_input, audio_input, movement_input])
encoded = layers.Dense(LATENT_DIM, activation="relu")(merged)

# **Decoder**
betas_output = layers.Dense(10, activation="tanh", name="betas")(encoded)
pose_output = layers.Dense(45, activation="tanh", name="pose")(encoded)

# **Final model**
model = tf.keras.Model([text_input, audio_input, movement_input], [betas_output, pose_output])
model.compile(optimizer="adam", loss="mse")

model_output_path = os.path.join(generated_file_folder_path, "models/encoder_decoder.h5")
model.save(model_output_path)

print("Model saved successfully.")
