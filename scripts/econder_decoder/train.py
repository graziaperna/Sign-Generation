import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

#nohup python -u scripts/feature_extraction/enc_dec.py > out.log 2>&1 &


def resample_sequence(data, target_length):
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    resized = F.interpolate(data_tensor, size=target_length, mode='area')
    resized = resized.transpose(1, 2).squeeze(0)
    return resized

def split_and_resample_motion(motion_array, num_samples, target_length, dim):
    total_frames = motion_array.shape[0]
    frames_per_sample = total_frames / num_samples
    segments = []
    for i in range(num_samples):
        start = int(i * frames_per_sample)
        end = int((i + 1) * frames_per_sample)
        segment = motion_array[start:end]
        if len(segment.shape) == 1:
            segment = np.expand_dims(segment, axis=-1)
        if segment.shape[-1] != dim:
            if segment.shape[-1] < dim:
                repeat = dim - segment.shape[-1]
                segment = np.concatenate([segment, np.repeat(segment[:, -1:], repeat, axis=-1)], axis=-1)
            else:
                segment = segment[:, :dim]
        resampled = resample_sequence(segment, target_length).cpu().numpy()
        segments.append(resampled)
    return np.stack(segments)


class RepeatText(nn.Module):
    def forward(self, text_state, audio_seq):
        time_steps = audio_seq.size(1)
        text_state_expanded = text_state.unsqueeze(1).repeat(1, time_steps, 1)
        return text_state_expanded

class Generator(nn.Module):
    def __init__(self, text_embedding_dim, audio_embedding_dim, latent_dim=16):
        super(Generator, self).__init__()
        self.text_encoder = nn.GRU(input_size=text_embedding_dim, hidden_size=latent_dim, batch_first=True)
        self.audio_encoder = nn.GRU(input_size=audio_embedding_dim, hidden_size=latent_dim, batch_first=True)
        self.decoder_gru = nn.GRU(input_size=latent_dim*2, hidden_size=latent_dim, batch_first=True)
        self.fc_betas = nn.Linear(latent_dim, 10)
        self.fc_rot_mats = nn.Linear(latent_dim, 42)
        self.repeat_text = RepeatText()
        
    def forward(self, text_input, audio_input):
        text_encoded, text_state = self.text_encoder(text_input)  
        text_state = text_state.squeeze(0)
        audio_encoded, _ = self.audio_encoder(audio_input)
        repeated_text = self.repeat_text(text_state, audio_encoded)
        x = torch.cat([audio_encoded, repeated_text], dim=-1)
        x, _ = self.decoder_gru(x)
        betas = self.fc_betas(x)
        rot_mats = self.fc_rot_mats(x)
        return betas, rot_mats

class Discriminator(nn.Module):
    def __init__(self, betas_dim=10, rot_dim=42):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=betas_dim+rot_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, betas, rot_mats):
        x = torch.cat([betas, rot_mats], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        features = x
        x = x.mean(dim=2)
        score = self.fc(x)
        return score, features

class GTMapper(nn.Module):
   
    def __init__(self, in_dim, out_dim):
        super(GTMapper, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

def dtw_distance_average(seq1, seq2):
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return distance / len(path)

def compute_velocity(seq):
    return np.diff(seq, axis=0)

def compute_acceleration(seq):
    v = compute_velocity(seq)
    return np.diff(v, axis=0)

def mse(a, b):
    return np.mean((a - b)**2)

def evaluate_geometric_errors(real_seq, fake_seq):
    mse_pose = mse(real_seq, fake_seq)
    real_vel = compute_velocity(real_seq)
    fake_vel = compute_velocity(fake_seq)
    mse_vel = mse(real_vel, fake_vel)
    real_acc = compute_acceleration(real_seq)
    fake_acc = compute_acceleration(fake_seq)
    mse_acc = mse(real_acc, fake_acc)
    return mse_pose, mse_vel, mse_acc

def diversity_metric(fake_sequences, num_pairs=20):
    n = len(fake_sequences)
    if n < 2:
        return 0.0
    distances = []
    for _ in range(num_pairs):
        i, j = random.sample(range(n), 2)
        seq1 = fake_sequences[i].reshape(-1)
        seq2 = fake_sequences[j].reshape(-1)
        dist = np.linalg.norm(seq1 - seq2)
        distances.append(dist)
    return np.mean(distances)

import torch.nn as nn

class SignBlueLoss(nn.Module):
    def __init__(self, weights=None):
        super(SignBlueLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

        if weights is None:
            self.weights = {
                "MSE_pose": 0.5,
                "MSE_vel": 0.2,
                "MSE_acc": 0.2,
                "DTW_mean": 0.2,
                "Diversity": 0.5
            }
        else:
            self.weights = weights

    def forward(self, real_betas, real_rots, fake_betas, fake_rots):

        mse_pose = self.mse_loss(fake_betas, real_betas) + self.mse_loss(fake_rots, real_rots)

        mse_vel = self.mse_loss(torch.diff(fake_betas, dim=1), torch.diff(real_betas, dim=1)) + \
                  self.mse_loss(torch.diff(fake_rots, dim=1), torch.diff(real_rots, dim=1))

        mse_acc = self.mse_loss(torch.diff(fake_betas, n=2, dim=1), torch.diff(real_betas, n=2, dim=1)) + \
                  self.mse_loss(torch.diff(fake_rots, n=2, dim=1), torch.diff(real_rots, n=2, dim=1))

        diversity_real_betas = torch.std(real_betas, dim=0).mean()
        diversity_fake_betas = torch.std(fake_betas, dim=0).mean()
        diversity_real_rots = torch.std(real_rots, dim=0).mean()
        diversity_fake_rots = torch.std(fake_rots, dim=0).mean()

        diversity_score = torch.abs(diversity_real_betas - diversity_fake_betas) + \
                          torch.abs(diversity_real_rots - diversity_fake_rots)

        signblue_loss = (
            self.weights["MSE_pose"] * mse_pose +
            self.weights["MSE_vel"] * mse_vel +
            self.weights["MSE_acc"] * mse_acc +
            self.weights["Diversity"] * diversity_score
        )

        return signblue_loss


def evaluate_motion_metrics(real_betas_epoch, real_rots_epoch,
                            fake_betas_epoch, fake_rots_epoch,
                            max_samples=20, downsample_factor=8):
    N = real_betas_epoch.shape[0]
    if N > max_samples:
        indices = random.sample(range(N), k=max_samples)
    else:
        indices = list(range(N))
        
    dtw_vals = []
    mse_pose_vals = []
    mse_vel_vals = []
    mse_acc_vals = []
    merged_fake = []
    
    for i in indices:
        rb = real_betas_epoch[i]
        rr = real_rots_epoch[i]
        fb = fake_betas_epoch[i]
        fr = fake_rots_epoch[i]
        
        rb_down = rb[::downsample_factor]
        rr_down = rr[::downsample_factor]
        fb_down = fb[::downsample_factor]
        fr_down = fr[::downsample_factor]
        
        dtw_betas = dtw_distance_average(rb_down, fb_down)
        dtw_rot   = dtw_distance_average(rr_down, fr_down)
        dtw_vals.append((dtw_betas + dtw_rot) / 2.0)
        
        if rb_down.shape[0] == fb_down.shape[0] and rr_down.shape[0] == fr_down.shape[0]:
            mse_b_pose, mse_b_vel, mse_b_acc = evaluate_geometric_errors(rb_down, fb_down)
            mse_r_pose, mse_r_vel, mse_r_acc = evaluate_geometric_errors(rr_down, fr_down)
            mse_pose_vals.append((mse_b_pose + mse_r_pose)/2.0)
            mse_vel_vals.append((mse_b_vel + mse_r_vel)/2.0)
            mse_acc_vals.append((mse_b_acc + mse_r_acc)/2.0)
        
        merged = np.concatenate([fb, fr], axis=-1)
        merged_fake.append(merged)
        
    dtw_mean = np.mean(dtw_vals) if dtw_vals else 0.0
    mse_pose_mean = np.mean(mse_pose_vals) if mse_pose_vals else 0.0
    mse_vel_mean  = np.mean(mse_vel_vals)  if mse_vel_vals else 0.0
    mse_acc_mean  = np.mean(mse_acc_vals)  if mse_acc_vals else 0.0
    merged_fake = np.array(merged_fake)
    div_value = diversity_metric(merged_fake, num_pairs=20) if len(merged_fake) > 1 else 0.0
    
    return {
        "DTW_mean": dtw_mean,
        "MSE_pose": mse_pose_mean,
        "MSE_vel": mse_vel_mean,
        "MSE_acc": mse_acc_mean,
        "Diversity": div_value
    }

def gradient_penalty(discriminator, real_betas, real_rots, fake_betas, fake_rots):
    batch_size = real_betas.size(0)
    alpha_betas = torch.rand(batch_size, 1, 1, device=real_betas.device)
    alpha_rot = torch.rand(batch_size, 1, 1, device=real_betas.device)
    interpolated_betas = alpha_betas * real_betas + (1 - alpha_betas) * fake_betas
    interpolated_rots = alpha_rot * real_rots + (1 - alpha_rot) * fake_rots
    interpolated_betas.requires_grad_(True)
    interpolated_rots.requires_grad_(True)
    
    interpolated_score, _ = discriminator(interpolated_betas, interpolated_rots)
    grad_outputs = torch.ones_like(interpolated_score, device=real_betas.device)
    gradients = torch.autograd.grad(
        outputs=interpolated_score,
        inputs=[interpolated_betas, interpolated_rots],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    grad_betas, grad_rots = gradients
    grad_betas = grad_betas.reshape(batch_size, -1)
    grad_rots = grad_rots.reshape(batch_size, -1)
    slopes = torch.sqrt(torch.sum(grad_betas**2, dim=1) + torch.sum(grad_rots**2, dim=1))
    gp = torch.mean((slopes - 1.0) ** 2)
    return gp


class MotionDataset(Dataset):
    def __init__(self, text_data, audio_data, betas_data, rots_data):
        self.text_data = text_data
        self.audio_data = audio_data
        self.betas_data = betas_data
        self.rots_data = rots_data
        
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_data[idx], dtype=torch.float32)
        audio = torch.tensor(self.audio_data[idx], dtype=torch.float32)
        betas = torch.tensor(self.betas_data[idx], dtype=torch.float32)
        rots = torch.tensor(self.rots_data[idx], dtype=torch.float32)
        return text, audio, betas, rots

def collate_fn(batch):
    texts, audios, betas, rots = zip(*batch)
    
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    audios = torch.stack(audios)
    betas = torch.stack(betas)
    rots = torch.stack(rots)
    return texts_padded, audios, betas, rots


def train_gan(generator, discriminator, 
              gt_beta_mapper, gt_rot_mapper,
              dataloader, epochs=200, gen_updates=2, csv_path='gan_metrics.csv'):
    
    gen_optimizer = optim.Adam(list(generator.parameters()) +
                               list(gt_beta_mapper.parameters()) +
                               list(gt_rot_mapper.parameters()),
                               lr=2e-4, betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=5e-6, betas=(0.5, 0.9))
    
    mse_loss = nn.MSELoss()
    metrics_history = []
    best_metric = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_d_loss = []
        epoch_g_loss = []
        real_betas_epoch = []
        real_rots_epoch = []
        fake_betas_epoch = []
        fake_rots_epoch = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device ", device)

        for text_batch, audio_batch, betas_batch, rots_batch in dataloader:
            torch.autograd.set_detect_anomaly(True)
            mapped_betas = gt_beta_mapper(betas_batch)
            mapped_rots = gt_rot_mapper(rots_batch)

            fake_betas, fake_rots = generator(text_batch, audio_batch)

            real_score, real_features = discriminator(mapped_betas, mapped_rots)
            fake_score, fake_features = discriminator(fake_betas, fake_rots)
            
            d_loss = fake_score.mean() - real_score.mean()
            gp = gradient_penalty(discriminator, mapped_betas, mapped_rots, fake_betas, fake_rots)
            d_loss = d_loss + 8.0 * gp
            
            disc_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            disc_optimizer.step()
            
            mapped_betas = mapped_betas.detach()
            mapped_rots  = mapped_rots.detach()
            real_features = real_features.detach()
            signblue_loss_fn = SignBlueLoss()

            for _ in range(gen_updates):
                # fake_betas, fake_rots = generator(text_batch, audio_batch)
                # fake_score, fake_features = discriminator(fake_betas, fake_rots)
                
                # g_loss = -fake_score.mean()
                # output_loss = mse_loss(fake_betas, target_betas) + mse_loss(fake_rots, target_rots)
                # fm_loss = mse_loss(real_features_detached, fake_features)
                # g_loss = g_loss + output_loss + fm_loss
                
                # gen_optimizer.zero_grad()
                # g_loss.backward()
                # gen_optimizer.step()
                fake_betas, fake_rots = generator(text_batch, audio_batch)
                fake_score, fake_features = discriminator(fake_betas, fake_rots)

                g_loss = -fake_score.mean() * 0.01

                signblue_loss = signblue_loss_fn(mapped_betas, mapped_rots, fake_betas, fake_rots)

                total_g_loss = g_loss + signblue_loss
                gen_optimizer.zero_grad()
                total_g_loss.backward()
                gen_optimizer.step()

            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(total_g_loss.item())
            real_betas_epoch.append(mapped_betas.detach().cpu().numpy())
            real_rots_epoch.append(mapped_rots.detach().cpu().numpy())
            fake_betas_epoch.append(fake_betas.detach().cpu().numpy())
            fake_rots_epoch.append(fake_rots.detach().cpu().numpy())
        
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        real_betas_epoch = np.concatenate(real_betas_epoch, axis=0)
        real_rots_epoch = np.concatenate(real_rots_epoch, axis=0)
        fake_betas_epoch = np.concatenate(fake_betas_epoch, axis=0)
        fake_rots_epoch = np.concatenate(fake_rots_epoch, axis=0)
        
        metrics_eval = evaluate_motion_metrics(real_betas_epoch, real_rots_epoch,
                                               fake_betas_epoch, fake_rots_epoch)
        
        print(f"Epoch {epoch+1} - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
              f"DTW_mean: {metrics_eval['DTW_mean']:.4f}, MSE_pose: {metrics_eval['MSE_pose']:.4f}, "
              f"MSE_vel: {metrics_eval['MSE_vel']:.4f}, MSE_acc: {metrics_eval['MSE_acc']:.4f}, "
              f"Diversity: {metrics_eval['Diversity']:.4f}\n")
        
        metrics_history.append({
            "epoch": epoch+1,
            "D_loss": avg_d_loss,
            "G_loss": avg_g_loss,
            "DTW_mean": metrics_eval['DTW_mean'],
            "MSE_pose": metrics_eval['MSE_pose'],
            "MSE_vel": metrics_eval['MSE_vel'],
            "MSE_acc": metrics_eval['MSE_acc'],
            "Diversity": metrics_eval['Diversity']
        })
        
        current_metric = metrics_eval['DTW_mean']
        if current_metric < best_metric:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}: DTW_mean did not improve for {patience} consecutive epochs.")
                break
    
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ["epoch", "D_loss", "G_loss", "DTW_mean", "MSE_pose", "MSE_vel", "MSE_acc", "Diversity"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_history:
            writer.writerow(m)
    
    print("Training finished. Metrics saved in ", csv_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
audio_embedding_path = os.path.join(generated_file_folder_path, "models/audio_embeddings.npy")
text_embedding_path = os.path.join(generated_file_folder_path, "models/text_embeddings.npy")
motion_path = os.path.join(generated_file_folder_path, "pkl_precessed.npz")
metrics_path = os.path.join(generated_file_folder_path, "models/gan_metrics.csv")

print("Loading FastText embeddings...")
text_embeddings = np.load(text_embedding_path)
print(f"text_embeddings shape: {text_embeddings.shape}")

print("Loading audio embeddings...")
audio_embeddings = np.load(audio_embedding_path)
print(f"audio_embeddings shape: {audio_embeddings.shape}")

print("Loading motions...")
motion_data = np.load(motion_path)
all_rot_mats = np.array(motion_data["all_rot_mats"], copy=True)
all_betas = np.array(motion_data["all_betas"], copy=True)
print(f"all_betas shape: {all_betas.shape}")
print(f"all_rot_mats shape: {all_rot_mats.shape}")

num_samples = audio_embeddings.shape[0]
target_length = audio_embeddings.shape[1]

all_betas_resampled = split_and_resample_motion(all_betas, num_samples, target_length, dim=1)
all_rot_mats_resampled = split_and_resample_motion(all_rot_mats, num_samples, target_length, dim=1)

n_audio = audio_embeddings.shape[0]
total_text = text_embeddings.shape[0]
base = total_text // n_audio 
remainder = total_text % n_audio
text_counts = [base] * n_audio
for i in range(remainder):
    text_counts[i] += 1

grouped_text_data = []
start = 0
for count in text_counts:
    group = text_embeddings[start:start+count]
    grouped_text_data.append(group)
    start += count

target_length_text = 16000
resampled_text_data = []
for group in grouped_text_data:
    group_tensor = torch.tensor(group, dtype=torch.float32)
    concatenated = group_tensor.view(-1, group_tensor.shape[-1])

    resampled = resample_sequence(concatenated.cpu().numpy(), target_length_text)
    resampled_text_data.append(resampled.cpu().numpy())

final_text_embeddings = np.stack(resampled_text_data, axis=0)
print(f"final_text_embeddings shape: {final_text_embeddings.shape}")


text_embedding_dim = final_text_embeddings.shape[-1]
audio_embedding_dim = audio_embeddings.shape[-1]

print("Building generator...")
generator = Generator(text_embedding_dim, audio_embedding_dim, latent_dim=32)

print("Building discriminator...")
discriminator = Discriminator(betas_dim=10, rot_dim=42)

gt_beta_mapper = GTMapper(in_dim=all_betas_resampled.shape[-1], out_dim=10)
gt_rot_mapper = GTMapper(in_dim=all_rot_mats_resampled.shape[-1], out_dim=42)


dataset = MotionDataset(final_text_embeddings, audio_embeddings, all_betas_resampled, all_rot_mats_resampled)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


print("Starting training...")
train_gan(generator, discriminator, gt_beta_mapper, gt_rot_mapper, dataloader,
          epochs=200, gen_updates=2, csv_path=metrics_path)

gen_path = os.path.join(generated_file_folder_path, "models/generator_model.pth")
disc_path = os.path.join(generated_file_folder_path, "models/discriminator_model.pth")
os.makedirs(os.path.dirname(gen_path), exist_ok=True)
torch.save(generator.state_dict(), gen_path)
torch.save(discriminator.state_dict(), disc_path)
print("Generator and discriminator saved.")
