import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gensim.models import FastText
from tensorflow.keras.optimizers import Adam
import csv
import gc
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from econder_decoder import const
import h5py
from gensim.models import FastText
from utils.file import FileUtil
print(tf.__version__)

LATENT_DIM = 16

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")


def process_data(data, name=""):
    if data is None or len(data) == 0:
        print(f"Data for {name} is empty or None.")
        return np.array([])

    if isinstance(data, dict):
        processed_values = []
        for key, value in data.items():
            if isinstance(value, list):
                value = [tensor_to_numpy(item) if isinstance(item, torch.Tensor) else item for item in value]
            else:
                value = tensor_to_numpy(value)

            value = np.ravel(value)
            processed_values.append(value)

        data = np.concatenate(processed_values) if processed_values else np.array([])

    elif isinstance(data, list):
        data = np.concatenate([np.ravel(tensor_to_numpy(item)) for item in data])

    elif isinstance(data, np.ndarray):
        data = np.ravel(data)

    elif isinstance(data, torch.Tensor):
        data = tensor_to_numpy(data).ravel()

    else:
        raise TypeError(f"Data type not supported for {name}.")

    return data


def get_movement_data(movement_data):

    rot_mats = process_data(movement_data.get("rot_mats"), "rot_mats")
    betas = process_data(movement_data.get("betas"), "betas")

    if any(x is None or len(x) == 0 for x in [rot_mats, betas]):
        raise TypeError("Skipping data due to missing or empty values in 'rot_mats' or 'betas'.")

    return rot_mats, betas


def tensor_to_numpy(tensor):
    '''
    Converts a tensor to a numpy array.
    '''
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    return tensor

beta_list = []
rot_mats_list = []
files = []

for root, dirs, filenames in os.walk(FileUtil.DATASER_INTEGRATION_PATH):
    dirs.sort(key=FileUtil.extract_date)
    filenames = sorted([f for f in filenames if f.endswith('.pkl')], key=FileUtil.extract_date_and_number_from_filename)
    for filename in filenames:
        files.append(os.path.join(root, filename))

for i, file in enumerate(files):
    try:
        data = pd.read_pickle(file)
        rot_mats, betas = get_movement_data(data)
        rot_mats_list.append(rot_mats)
        beta_list.append(betas)
        print(f"File {file} loading succeded.")
    except Exception as e:
        print(f"Error loading {file}: {e}")

    print(f"File {file} processed.")
    print(f"Files remaining: {len(files) - (i + 1)}")

import psutil
print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
print("Data concatenation starting...")
all_rot_mats = np.concatenate(rot_mats_list, axis=0)
del rot_mats_list
all_betas = np.concatenate(beta_list, axis=0)
del beta_list
print("Data concatenation completed.")
gc.collect()
print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

if all_rot_mats.ndim == 1:
    all_rot_mats = np.expand_dims(all_rot_mats, axis=1)

if all_betas.ndim == 1:
    all_betas = np.expand_dims(all_betas, axis=1)

np.savez_compressed(os.path.join(generated_file_folder_path, "pkl_precessed.npz"), all_rot_mats=all_rot_mats, all_betas=all_betas)