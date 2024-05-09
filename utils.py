import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from matplotlib import pyplot as plt
import warnings

def project_and_filter(X, dir, percentile):
    projs = np.dot(X, dir)
    thresh = np.percentile(projs, 100 - percentile)
    filtered_idxs = projs >= thresh
    return X[filtered_idxs], filtered_idxs

def clust_filter(X, sub, clusters):
  """
  Args:
    X - (N, d) array of data
    sub - (k,) binary vector corresponding to a subset of clusters
    clusters - (N,) what cluster each datapoint belongs to
  Returns:
    X_sub - (S, d)
    idxs - (S,)
  """
  included_clusters = np.where(sub == 1)[0]
  idxs = np.isin(clusters, included_clusters)
  return X[idxs], idxs

# Filter out ConvergenceWarning for logistic regression
# Happens occasionally with non liblinear solver
# Dataset dependent
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def log_coeffs(X, Y, liblin):
    """
    Args:
    X: N x d matrix of input features
    Y: N x 1 matrix (column vector) of output response

    Returns:
    Beta: d+1 matrix of linear coefficients (first dim on last axis is bias dim)
    Acc: Accuracy of model on training data
    """
    acc = None
    if liblin: logreg = LogisticRegression(random_state=0, solver='liblinear').fit(X, Y)
    else: logreg = LogisticRegression(random_state=0).fit(X, Y)

    betas = np.hstack((logreg.intercept_[:,None], logreg.coef_))

    ext_X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=-1)
    prd = (1 / (1 + np.exp(- ext_X @ betas.T)) > 0.5).astype(int)
    acc = np.mean(prd[:, 0]==Y)
    return np.squeeze(betas), acc

def get_preds(X, betas):
    """
    Args:
    randfeats: N x d
    betas: M x d+1
    Return:
    preds: N x M - each beta predicts on each instance
    """
    betas = np.array(betas)
    ext_X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    prds = 1 / (1 + np.exp(-ext_X @ betas.T))
    return prds

def aggregate_preds(preds):
    """
    Args:
      preds - (n, M) where M is num of models in ensemble
        (Can be one in directional, so need to check)
    """
    if len(preds.shape) == 1:
        preds = tf.squeeze(preds)
        return np.float32(preds > 0.5), None, None
    mean_pred = np.mean(preds, axis=-1, keepdims=False)
    std_pred = np.std(preds, axis=-1, keepdims=False)
    return (mean_pred > 0.5).astype(int), np.float32(mean_pred), np.float32(std_pred)

def evaluate(X, Y, betas):
    prds = get_preds(X, betas)
    preds_bin, mean_pred, std_pred = aggregate_preds(prds)
    acc = sum(preds_bin == Y) / len(Y)
    print("Accuracy: ", acc)
    print("Precision: ", precision_score(Y, preds_bin))
    print("Recall: ", recall_score(Y, preds_bin))

    precision, recall, thresholds = precision_recall_curve(Y, preds_bin)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('IID-Val Precision-Recall Curve')
    plt.show()