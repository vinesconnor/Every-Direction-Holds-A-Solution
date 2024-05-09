import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CosineSimilarity
from sklearn.model_selection import train_test_split
from utils import get_preds, clust_filter

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

def get_subset_agreement(clust_test, beta_test, modeled_betas, X_rand, labels):
    agg = []
    for clust, beta, beta_hat in zip(clust_test, beta_test, modeled_betas):
        _, X_ids = clust_filter(X_rand, clust, labels)
        X_sub = X_rand[X_ids]
        true_preds = get_preds(X_sub, beta)
        modeled_preds = get_preds(X_sub, beta_hat)
        agg.append(get_agreement(true_preds, modeled_preds))
    agg = np.mean(agg)
    return agg

def generate(mlp, M, c):
    gen_clusters = np.random.randn(M, c)
    gen_clusters = gen_clusters / np.linalg.norm(gen_clusters, axis=1, keepdims=True)
    gen_betas = mlp.predict(gen_clusters)
    return gen_clusters, gen_betas

def mlp_main(clusts, betas, X_pca, X_rand, labels):
    # Split the data into training and testing sets
    clust_train, clust_test, beta_train, beta_test = train_test_split(clusts, betas, test_size=0.2, random_state=42)
    
    mlp = create_mlp(clust_train.shape[-1], beta_train.shape[-1])
    mlp.fit(clust_train, beta_train, epochs=10)

    # Get basic test loss
    test_loss = mlp.evaluate(clust_test, beta_test)
    modeled_betas = mlp.predict(clust_test)

    # Get agreement loss
    true_preds = get_preds(X_rand, beta_test)
    modeled_preds = get_preds(X_rand, modeled_betas)
    overall_agreement = get_agreement(true_preds, modeled_preds)

    # Get subset agreement
    subset_agreement = get_subset_agreement(clust_test, beta_test, modeled_betas, X_rand, labels)

    return {
        'mlp': mlp,
        'test_loss': test_loss,
        'overall_agreement': overall_agreement,
        'subset_agreement': subset_agreement
    }