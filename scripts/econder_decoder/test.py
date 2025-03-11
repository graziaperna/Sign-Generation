from tensorflow.keras.models import load_model
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import torch.nn as nn


from feature_extraction import predict_text_embeddings
from feature_extraction import predict_audio_embeddings

current_dir = os.path.dirname(os.path.abspath(__file__))

generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
xlsx_file_folder_path = os.path.join(current_dir, "../../Tg_Noi_Lis_01_06_2022.txt")
audio_file_folder_path = os.path.join(current_dir, "../../Tg_Noi_Lis_01_06_2022.mp3")

gen_path = os.path.join(generated_file_folder_path, "models/generator_model_5.pth")
audio_embedding_path = os.path.join(generated_file_folder_path, "embeddings/audio_sample.mp3")


class RepeatText(nn.Module):
    def forward(self, text_state, audio_encoded):
        repeated_text = text_state.unsqueeze(1).expand(-1, audio_encoded.size(1), -1)
        return repeated_text

class Generator(nn.Module):
    def __init__(self, text_embedding_dim, audio_embedding_dim, latent_dim=16, noise_dim=8):
        super(Generator, self).__init__()
        self.text_encoder = nn.GRU(input_size=text_embedding_dim, hidden_size=latent_dim, batch_first=True)
        self.audio_encoder = nn.GRU(input_size=audio_embedding_dim, hidden_size=latent_dim, batch_first=True)
        self.decoder_gru = nn.GRU(input_size=(latent_dim * 2) + noise_dim, hidden_size=latent_dim, batch_first=True)
        #self.decoder_gru = nn.GRU(input_size=(latent_dim * 2), hidden_size=latent_dim, batch_first=True)
        self.fc_betas = nn.Linear(latent_dim, 10)
        self.fc_rot_mats = nn.Linear(latent_dim, 42)
        self.repeat_text = RepeatText()
        self.noise_dim = noise_dim
        
    def forward(self, text_input, audio_input):
        text_encoded, text_state = self.text_encoder(text_input)  
        text_state = text_state.squeeze(0)
        audio_encoded, _ = self.audio_encoder(audio_input)
        repeated_text = self.repeat_text(text_state, audio_encoded)
        latent_noise = torch.randn((audio_encoded.shape[0], audio_encoded.shape[1], self.noise_dim), device=text_input.device)
        x = torch.cat([audio_encoded, repeated_text, latent_noise], dim=-1)
        #x = torch.cat([audio_encoded, repeated_text], dim=-1)
        x, _ = self.decoder_gru(x)
        betas = self.fc_betas(x)
        rot_mats = self.fc_rot_mats(x)
        return betas, rot_mats

    def predict(self, text_input, audio_input):
        self.eval()
        with torch.no_grad():
            return self.forward(text_input, audio_input)

latent_dim = 32

df = pd.read_csv(xlsx_file_folder_path, sep=" -- ", header=None, encoding="latin1", engine="python")
df.columns = ['start_time', 'end_time', 'text']
print(df["text"])
text_embeddings = predict_text_embeddings.get_text_embeddings(" ".join(df["text"].astype(str)))
print("Generated embedding for input text.")

audio_embeddings = predict_audio_embeddings.getAudioEmbedding(audio_file_folder_path)
print("Generated embedding for input audio.")

print(f"Initial text_embeddings shape: {text_embeddings.shape}")
print(f"Initial audio_embeddings shape: {audio_embeddings.shape}")

text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
audio_embeddings = torch.tensor(audio_embeddings, dtype=torch.float32)

model = Generator(text_embeddings.shape[2], audio_embeddings.shape[2], latent_dim)

state_dict = torch.load(gen_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

betas, rot_mats = model.predict(text_embeddings, audio_embeddings)

output_dir = os.path.join(generated_file_folder_path, "models/embeddings/")
os.makedirs(output_dir, exist_ok=True)
print("Generated betas:", betas)
print("Generated rotation matrices:", rot_mats)
np.save(os.path.join(output_dir, "generated_betas.npy"), betas)
np.save(os.path.join(output_dir, "generated_rot_mats.npy"), rot_mats)

txt_path = os.path.join(output_dir, "betas_and_rot.txt")
with open(txt_path, "w") as f:
    f.write("Betas:\n")
    betas_2d = betas.reshape(-1, betas.shape[-1])
    np.savetxt(f, betas_2d, fmt="%.6f", delimiter=" ")
    
    f.write("\nRotation Matrices:\n")
    rot_2d = rot_mats.reshape(-1, rot_mats.shape[-1])
    np.savetxt(f, rot_2d, fmt="%.6f", delimiter=" ")
    
print("File generated:", txt_path)