import pickle
import os
import numpy as np
import torch


def tensor_to_numpy(tensor):
    """Converte un tensore PyTorch in un array NumPy, spostandolo prima su CPU se necessario."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def process_data(data, name=""):
    """Processa i dati, convertendo tensori in array NumPy e appiattendo il risultato."""
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
        raise TypeError(f"Data type not supported for {name}.")

    return data


movement_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets/integration/Tg_Noi_Lis_15_04_2022/Tg_Noi_Lis_15_04_2022_0.pkl")

# Carica il file pickle
with open(movement_model_path, "rb") as f:
    movement_data = pickle.load(f)

print("movement keys:", movement_data.keys())

processed_data = {}
for key, value in movement_data.items():
    try:
        # Processa i dati per ogni chiave
        processed_data[key] = process_data(value, name=key)
        print(f"{key}: {type(processed_data[key])} - Shape: {processed_data[key].shape if isinstance(processed_data[key], np.ndarray) else 'N/A'}")
        
        # Controlla se 'rot_mats' è presente
        if 'rot_mats' in processed_data:
            print("Found rot_mats:", processed_data['rot_mats'])
        else:
            print("rot_mats not found in the file.")
        
    except Exception as e:
        print(f"Error in processing {key}: {e}")
