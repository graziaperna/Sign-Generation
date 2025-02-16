import os
import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import FastText
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import csv
import sys
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from econder_decoder import const
import h5py
from gensim.models import FastText
from utils.file import FileUtil
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
from tensorflow.keras import layers, Model

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_file_folder_path = os.path.join(current_dir, "../../generated_file/")
audio_model_path = os.path.join(generated_file_folder_path, "models/audio_embedding.h5")
text_model_path = os.path.join(generated_file_folder_path, "models/fasttext_model.model")
text_embedding_path = os.path.join(generated_file_folder_path, "models/fasttext_word_embeddings.npy")
metrics_path = os.path.join(generated_file_folder_path, "models/gan_metrics.csv")

tf.keras.backend.clear_session()
gc.collect()

print("Loading FastText model...")
ft_model = FastText.load(text_model_path)
text_embeddings = np.load(text_embedding_path)

print(f"text_embeddings shape: {text_embeddings.shape}")

print("Loading audio embeddings...")
with h5py.File(audio_model_path, "r") as f:
    audio_embeddings = np.array(f[const.AUDIO_EMBEDDING_NAME])
print(f"audio_embeddings shape: {audio_embeddings.shape}")

print("Loading motions...")
motion_data = np.load(os.path.join(generated_file_folder_path, "pkl_precessed.npz"))
all_rot_mats = np.array(motion_data["all_rot_mats"], copy=True)
all_betas = np.array(motion_data["all_betas"], copy=True)
print(f"all_betas : {all_betas.shape}")
print(f"all_rot_mats : {all_rot_mats.shape}")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import csv

# -------------------------
# FUNZIONI DI RESAMPLING (rimangono invariate)
# -------------------------

def resample_sequence(data, target_length):
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    data_tensor = tf.expand_dims(data_tensor, axis=0)
    data_tensor = tf.expand_dims(data_tensor, axis=-1)
    resized = tf.image.resize(data_tensor, size=(target_length, tf.shape(data_tensor)[2]), method='area')
    resized = tf.squeeze(resized, axis=0)
    resized = tf.squeeze(resized, axis=-1)
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
        resampled = resample_sequence(segment, target_length)
        segments.append(resampled.numpy())
    return np.stack(segments)

# -------------------------
# MODELLI
# -------------------------

def build_generator(text_embedding_dim, audio_embedding_dim, latent_dim=32):
    text_input = layers.Input(shape=(None, text_embedding_dim), name='text_input')
    audio_input = layers.Input(shape=(None, audio_embedding_dim), name='audio_input')
    
    x_text = layers.Masking(mask_value=0.0)(text_input)
    x_audio = layers.Masking(mask_value=0.0)(audio_input)
    
    text_encoded = layers.GRU(latent_dim, return_sequences=True, activation='tanh', name='text_encoder')(x_text)
    audio_encoded = layers.GRU(latent_dim, return_sequences=True, activation='tanh', name='audio_encoder')(x_audio)
    
    x = layers.Concatenate(axis=-1, name='concat_inputs')([text_encoded, audio_encoded])
    
    x = layers.GRU(latent_dim, return_sequences=True, activation='tanh', name='decoder_gru')(x)
    
    betas = layers.TimeDistributed(layers.Dense(10, activation='linear'), name='betas_output')(x)
    rot_mats = layers.TimeDistributed(layers.Dense(9, activation='linear'), name='rot_mats_output')(x)
    
    return Model(inputs=[text_input, audio_input],
                 outputs=[betas, rot_mats],
                 name='generator')

def build_discriminator(betas_dim=10, rot_dim=9):
    betas_input = layers.Input(shape=(None, betas_dim), name='betas_input')
    rot_input = layers.Input(shape=(None, rot_dim), name='rot_mats_input')
    
    x = layers.Concatenate(axis=-1, name='concat_discriminator')([betas_input, rot_input])
    x = layers.Conv1D(32, kernel_size=3, activation=None, padding='same', name='disc_conv')(x)
    x = layers.BatchNormalization(name='disc_bn')(x)
    x = layers.LeakyReLU(alpha=0.2, name='disc_leakyrelu')(x)
    x = layers.Dropout(0.1, name='disc_dropout')(x)  # Riduce la capacità del discriminatore
    
    # Salviamo le feature intermedie per feature matching
    features = x
    x = layers.GlobalAveragePooling1D(name='disc_gap')(x)
    score = layers.Dense(1, activation='linear', name='disc_score')(x)
    
    model = Model(inputs=[betas_input, rot_input],
                  outputs=[score, features],
                  name='discriminator')
    return model


# -------------------------
# TERMINE DI GRADIENT PENALTY
# -------------------------

def gradient_penalty(discriminator, real, fake):
    batch_size = tf.shape(real[0])[0]  # usa il batch size dal primo tensore
    alpha_betas = tf.random.uniform(shape=tf.shape(real[0]), minval=0., maxval=1.)
    alpha_rot = tf.random.uniform(shape=tf.shape(real[1]), minval=0., maxval=1.)
    interpolated_betas = alpha_betas * real[0] + (1 - alpha_betas) * fake[0]
    interpolated_rot = alpha_rot * real[1] + (1 - alpha_rot) * fake[1]

    with tf.GradientTape() as tape:
        tape.watch([interpolated_betas, interpolated_rot])
        interp_score, _ = discriminator([interpolated_betas, interpolated_rot], training=True)
    grads_betas, grads_rot = tape.gradient(interp_score, [interpolated_betas, interpolated_rot])
    grads = tf.concat([tf.reshape(grads_betas, [batch_size, -1]),
                       tf.reshape(grads_rot, [batch_size, -1])], axis=1)
    gp = tf.reduce_mean((tf.norm(grads, axis=1) + 1e-8 - 1.)**2)
    return gp


def train_gan(generator, discriminator,
              text_data, audio_data, real_betas, real_rot_mats,
              epochs=1000, batch_size=128, csv_path='gan_metrics.csv',
              gen_updates=2):

    text_data = tf.convert_to_tensor(text_data, dtype=tf.float32)
    audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    real_betas = tf.convert_to_tensor(real_betas, dtype=tf.float32)
    real_rot_mats = tf.convert_to_tensor(real_rot_mats, dtype=tf.float32)
    
    if len(text_data.shape) == 2:
        text_data = tf.expand_dims(text_data, axis=1)
    if len(audio_data.shape) == 2:
        audio_data = tf.expand_dims(audio_data, axis=1)
    if len(real_betas.shape) == 2:
        real_betas = tf.expand_dims(real_betas, axis=1)
    if len(real_rot_mats.shape) == 2:
        real_rot_mats = tf.expand_dims(real_rot_mats, axis=1)
    
    dataset = tf.data.Dataset.from_tensor_slices((text_data, audio_data, real_betas, real_rot_mats))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Imposto learning rate differenti:
    # Il generatore ha un learning rate più alto
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=6e-6, beta_1=0.5, beta_2=0.9)

    
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    # Mapper per i ground truth (rimangono invariati)
    gt_beta_mapper = layers.Dense(10, activation=None, name='gt_beta_mapper')
    gt_rot_mapper = layers.Dense(9, activation=None, name='gt_rot_mapper')
    
    metrics_history = []
    
    @tf.function
    def train_step(text_batch, audio_batch, betas_batch, rot_mats_batch):

        with tf.GradientTape(persistent=True) as tape:
            mapped_betas = gt_beta_mapper(betas_batch)
            mapped_rot_mats = gt_rot_mapper(rot_mats_batch)
            
            fake_betas, fake_rot_mats = generator([text_batch, audio_batch], training=True)
            
            real_score, real_features = discriminator([mapped_betas, mapped_rot_mats], training=True)
            fake_score, fake_features = discriminator([fake_betas, fake_rot_mats], training=True)
            
            # Loss del discriminatore (WGAN)
            d_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
            # Gradient Penalty per il discriminatore
            gp = gradient_penalty(discriminator, (mapped_betas, mapped_rot_mats), (fake_betas, fake_rot_mats))
            lambda_gp = 10.0
            d_loss += lambda_gp * gp
            
            # Loss del generatore: 
            # - L'obiettivo WGAN
            g_loss = -tf.reduce_mean(fake_score)
            # - Output loss: confronto tra output e ground truth mappati
            output_loss = mse_loss(fake_betas, mapped_betas) + mse_loss(fake_rot_mats, mapped_rot_mats)
            # - Feature matching loss: cerca di far sì che le feature intermedie siano simili
            fm_loss = mse_loss(real_features, fake_features)
            # Aumentiamo il peso del feature matching loss
            g_loss += 0.1 * output_loss + 0.2 * fm_loss
        
        disc_vars = discriminator.trainable_variables
        gen_vars = generator.trainable_variables + gt_beta_mapper.trainable_variables + gt_rot_mapper.trainable_variables
        
        disc_grads = tape.gradient(d_loss, disc_vars)
        disc_optimizer.apply_gradients(zip(disc_grads, disc_vars))
        
        # Aggiornamento multiplo del generatore
        for _ in range(gen_updates):
            gen_grads = tape.gradient(g_loss, gen_vars)
            gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))
        del tape
        
        real_correct = tf.cast(real_score > 0, tf.float32)
        fake_correct = tf.cast(fake_score < 0, tf.float32)
        d_acc = (tf.reduce_mean(real_correct) + tf.reduce_mean(fake_correct)) / 2.0
        g_acc = tf.reduce_mean(tf.cast(fake_score > 0, tf.float32))
        
        return d_loss, g_loss, tf.reduce_mean(real_score), tf.reduce_mean(fake_score), d_acc, g_acc
    
    for epoch in range(epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_d_acc = []
        epoch_g_acc = []
        epoch_r_score = []
        epoch_f_score = []
        
        for text_batch, audio_batch, betas_batch, rot_mats_batch in dataset:
            d_loss, g_loss, r_score, f_score, d_acc, g_acc = train_step(text_batch, audio_batch, betas_batch, rot_mats_batch)
            epoch_d_loss.append(d_loss.numpy())
            epoch_g_loss.append(g_loss.numpy())
            epoch_d_acc.append(d_acc.numpy())
            epoch_g_acc.append(g_acc.numpy())
            epoch_r_score.append(r_score.numpy())
            epoch_f_score.append(f_score.numpy())
        
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        avg_d_acc = np.mean(epoch_d_acc)
        avg_g_acc = np.mean(epoch_g_acc)
        avg_r_score = np.mean(epoch_r_score)
        avg_f_score = np.mean(epoch_f_score)
        
        tf.print(f"Epoch {epoch+1}/{epochs} - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
                 f"Real_score: {avg_r_score:.4f}, Fake_score: {avg_f_score:.4f}, "
                 f"D_acc: {avg_d_acc:.4f}, G_acc: {avg_g_acc:.4f}")
        
        metrics_history.append({
            "epoch": epoch+1,
            "D_loss": avg_d_loss,
            "G_loss": avg_g_loss,
            "Real_score": avg_r_score,
            "Fake_score": avg_f_score,
            "D_acc": avg_d_acc,
            "G_acc": avg_g_acc
        })
    
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ["epoch", "D_loss", "G_loss", "Real_score", "Fake_score", "D_acc", "G_acc"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_history:
            writer.writerow(m)
    
    tf.print("Training terminato e metriche salvate in", csv_path)

# -------------------------
# MAIN: Caricamento dati e training
# -------------------------

print("Loading FastText model...")
print(f"text_embeddings shape: {text_embeddings.shape}")

print("Loading audio embeddings...")
print(f"audio_embeddings shape: {audio_embeddings.shape}")

print("Loading motions...")
print(f"all_betas shape: {all_betas.shape}")
print(f"all_rot_mats shape: {all_rot_mats.shape}")

num_samples = 128
target_length = 128

all_betas_resampled = split_and_resample_motion(all_betas, num_samples, target_length, dim=1)
all_rot_mats_resampled = split_and_resample_motion(all_rot_mats, num_samples, target_length, dim=1)

print("Resampled all_betas shape:", all_betas_resampled.shape)
print("Resampled all_rot_mats shape:", all_rot_mats_resampled.shape)

text_embedding_dim = text_embeddings.shape[-1]
audio_embedding_dim = audio_embeddings.shape[-1]

tf.print("Building generator...")
generator = build_generator(text_embedding_dim, audio_embedding_dim, latent_dim=32)
generator.summary()

tf.print("Building discriminator...")
discriminator = build_discriminator(betas_dim=10, rot_dim=9)
discriminator.summary()

tf.print("Starting training...")
train_gan(generator, discriminator,text_embeddings, audio_embeddings, all_betas_resampled, all_rot_mats_resampled)

gen_path = os.path.join(generated_file_folder_path, "generator_model.keras")
disc_path = os.path.join(generated_file_folder_path, "discriminator_model.keras")
os.makedirs(os.path.dirname(gen_path), exist_ok=True)
generator.export(gen_path)
discriminator.export(disc_path)
tf.print("Generator and discriminator saved.")
