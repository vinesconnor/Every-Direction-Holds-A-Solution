import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CosineSimilarity
from sklearn.model_selection import train_test_split
from utils import get_preds, project_and_filter

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

def create_vae(beta_dim, input_dir_dim, latent_dim):
    # Encoder
    beta_input = layers.Input(shape=(beta_dim,))
    beta_x = layers.Dense(512, activation=tf.nn.elu)(beta_input)
    dir_input = layers.Input(shape=(input_dir_dim,))
    encoder_inputs = layers.Concatenate()([beta_x, dir_input])
    # x = layers.Dense(1024, activation=tf.nn.elu)(encoder_inputs)
    x = layers.Dense(512, activation=tf.nn.elu)(encoder_inputs)
    # x = layers.Dense(256, activation=tf.nn.elu)(encoder_inputs)
    x = layers.Dense(128, activation=tf.nn.elu)(encoder_inputs)
    x = layers.Dense(32, activation=tf.nn.elu)(x)
    # x = layers.Dense(8, activation=tf.nn.elu)(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    decoder_dir_input = layers.Input(shape=(input_dir_dim,))
    decoder_inputs = layers.Concatenate()([latent_inputs, decoder_dir_input])
    x = layers.Dense(32, activation=tf.nn.elu)(decoder_inputs)
    x = layers.Dense(256, activation=tf.nn.elu)(x)
    x = layers.Dense(512, activation=tf.nn.elu)(x)
    beta_output = layers.Dense(beta_dim)(x)

    # Instantiate model
    encoder = models.Model([beta_input, dir_input], [z_mean, z_log_var, z], name="encoder")
    decoder = models.Model([latent_inputs, decoder_dir_input], beta_output, name="decoder")

    # VAE
    outputs = decoder([encoder([beta_input, dir_input])[2], dir_input])
    vae = models.Model([beta_input, dir_input], outputs, name="vae")
    vae.encoder = encoder
    vae.decoder = decoder

    return vae

def train_step(model, beta_inputs, dir_inputs, opt, reg=1.0):
  with tf.GradientTape() as tape:
    z_mean, z_log_var, z = model.encoder([beta_inputs, dir_inputs])
    dir_inputs = tf.constant(dir_inputs)
    outputs = model.decoder([z, dir_inputs])
    total_loss, recon_loss, kl_loss = vae_loss(beta_inputs, outputs, z_mean, z_log_var, reg)
  grads = tape.gradient(total_loss, model.trainable_variables)
  opt.apply_gradients(zip(grads, model.trainable_variables))
  return total_loss, recon_loss, kl_loss

def vae_loss(inputs, outputs, z_mean, z_log_var, reg=1.0):
  inputs = tf.cast(inputs, tf.float32)
  recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=-1))
  kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
  total_loss = recon_loss + reg * kl_loss
  return total_loss, recon_loss, kl_loss

def batch(betas, dirs, batch_size):
  num_samples = betas.shape[0]
  indices = np.arange(num_samples)
  np.random.shuffle(indices)
  betas = np.array(betas)[indices]
  dirs = np.array(dirs)[indices]
  for i in range(0, betas.shape[0], batch_size):
    yield betas[i:i+batch_size], dirs[i:i+batch_size]

def train(vae, betas, dirs, reg):
    opt = tf.keras.optimizers.Adam()
    epochs = 33
    batch_size = 32

    for i in range(epochs):
        print(f"Epoch {i}")
        for step, (batch_betas, batch_dirs) in enumerate(batch(betas, dirs, batch_size)):
            loss_vals = train_step(vae, batch_betas, batch_dirs, opt, reg)
            if step % 8 == 0: # tmp
                print(f"Step {step}: loss = {loss_vals[0].numpy()}, recon_loss = {loss_vals[1].numpy()}, kl_loss = {loss_vals[2].numpy()}")
        print()

def get_agreement(true_preds, modeled_preds):
    """
    Args:
        true_preds: N x M
        pred_preds: N x M
    Returns:
        accuracy : Accuracy with respect to the ground truth betas, not labels
    """
    true_preds = (true_preds > 0.5).astype(int)
    modeled_preds = (modeled_preds > 0.5).astype(int)
    return np.mean(true_preds == modeled_preds)

def get_subset_agreement(dir_test, beta_test, modeled_betas, X_pca, X_rand, percentile):
    agg = []
    for dir, beta, beta_hat in zip(dir_test, beta_test, modeled_betas):
        _, X_ids = project_and_filter(X_pca, dir, percentile)
        X_sub = X_rand[X_ids]
        true_preds = get_preds(X_sub, beta)
        modeled_preds = get_preds(X_sub, beta_hat)
        agg.append(get_agreement(true_preds, modeled_preds))
    agg = np.mean(agg)
    return agg

def generate(model, M, K, z):
    gen_dirs = np.random.randn(M, K)
    gen_dirs = gen_dirs / np.linalg.norm(gen_dirs, axis=1, keepdims=True)
    gen_dirs = tf.constant(gen_dirs)
    latent_samples = tf.random.normal(shape=(M, z))
    return gen_dirs, model.decoder([latent_samples, gen_dirs])

def vae_main(dirs, betas, z, X_pca, X_rand, percentile, reg):
    # Split the data into training and testing sets
    dir_train, dir_test, beta_train, beta_test = train_test_split(np.array(dirs), np.array(betas), test_size=0.2, random_state=42)
    dir_train = tf.constant(dir_train)
    dir_test = tf.constant(dir_test)
    beta_train = tf.constant(beta_train)
    beta_test = tf.constant(beta_test)
    vae = create_vae(beta_train.shape[-1], dir_train.shape[-1], z)
    train(vae, beta_train, dir_train, reg)

    # Get basic test loss
    test_loss, modeled_betas = eval(vae, dir_test, beta_test, reg)

    # Get agreement loss
    true_preds = get_preds(X_rand, beta_test)
    modeled_preds = get_preds(X_rand, modeled_betas)
    overall_agreement = get_agreement(true_preds, modeled_preds)

    # Get subset agreement
    subset_agreement = get_subset_agreement(dir_test, beta_test, modeled_betas, X_pca, X_rand, percentile)

    return {
        'vae': vae,
        'test_loss': test_loss,
        'overall_agreement': overall_agreement,
        'subset_agreement': subset_agreement
    }

def eval(vae, dirs, betas, reg):
   z_mean, z_log_var, z = vae.encoder([betas, dirs])
   out = vae.decoder([z, dirs])
   return vae_loss(betas, out, z_mean, z_log_var, reg), out
