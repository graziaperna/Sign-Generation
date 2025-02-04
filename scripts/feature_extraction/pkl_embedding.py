import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gc
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

from utils.file import FileUtil

def tensor_to_numpy(tensor):
    '''
    Converts a tensor to a numpy array.
    '''
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    return tensor

def normalize(data, name):
    '''
    It normalizes the data and converts it to a numpy array.
    '''
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
        print(f"File {file} loading succeded.")
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

    combined_data = np.concatenate(
        [joints_normalized, betas_normalized, phi_normalized,
         rot_mats_normalized, expression_normalized,
         camera_normalized, focal_normalized],
        axis=-1
    )

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

print(f"Shape of all_combined_data: {all_combined_data.shape}")

model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(all_combined_data.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu')
])

print("Compilation of the model starting...")
model.compile(optimizer='adam', loss='mse')
print("Compilation of the model finished.")

model_destination_path = FileUtil.get_subdirectory_file_path(os.getcwd(), FileUtil.MODEL_FOLDER_PATH, "movement_embedding.h5")
model.save(model_destination_path)
