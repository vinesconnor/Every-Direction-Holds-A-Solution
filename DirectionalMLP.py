import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CosineSimilarity
from sklearn.model_selection import train_test_split
from utils import get_preds, project_and_filter

def create_mlp(d_in, d_out):
    mlp = Sequential([
    Dense(256, activation='elu', input_shape=(d_in,)),  # First layer
    Dropout(0.5),  # Dropout for regularization
    Dense(512, activation='elu'),  # Second layer
    Dropout(0.5),  # Dropout for regularization
    Dense(d_out)  # Output layer matches dimensionality of betas
    ])

    # Compile the model
    mlp.compile(optimizer='adam', loss='mse')
    return mlp

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

def generate(mlp, M, K):
    gen_dirs = np.random.randn(M, K)
    gen_dirs = gen_dirs / np.linalg.norm(gen_dirs, axis=1, keepdims=True)
    gen_betas = mlp.predict(gen_dirs)
    return gen_dirs, gen_betas

def mlp_main(dirs, betas, X_pca, X_rand, percentile):
    # Split the data into training and testing sets
    dir_train, dir_test, beta_train, beta_test = train_test_split(dirs, betas, test_size=0.2, random_state=42)
    
    mlp = create_mlp(dir_train.shape[-1], beta_train.shape[-1])
    mlp.fit(dir_train, beta_train, epochs=10)

    # Get basic test loss
    test_loss = mlp.evaluate(dir_test, beta_test)
    modeled_betas = mlp.predict(dir_test)

    # Get agreement loss
    true_preds = get_preds(X_rand, beta_test)
    modeled_preds = get_preds(X_rand, modeled_betas)
    overall_agreement = get_agreement(true_preds, modeled_preds)

    # Get subset agreement
    subset_agreement = get_subset_agreement(dir_test, beta_test, modeled_betas, X_pca, X_rand, percentile)

    return {
        'mlp': mlp,
        'test_loss': test_loss,
        'overall_agreement': overall_agreement,
        'subset_agreement': subset_agreement
    }