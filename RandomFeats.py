import numpy as np
import tensorflow as tf

class RandFeats:
  def __init__(self, sigma_rot, K, D):

    self.sigmas = [sigma_rot/4, sigma_rot/2, sigma_rot, sigma_rot*2, sigma_rot*4]
    self.D = D
    self.Ws = []
    for sigma in self.sigmas:
      self.Ws.append(np.float32(np.random.randn(K, D)/sigma))
    self.Ws = np.stack(self.Ws, 0)

  def get_features(self, x_in):
    phis = tf.matmul(x_in, self.Ws)  # 5 x N x D
    phis = tf.transpose(phis, [1, 2, 0])  # N x D x 5
    phis = tf.concat((tf.sin(phis), tf.cos(phis)), 1) # N x D x 10
    return tf.reshape(phis, [x_in.shape[0], -1]) # N x 10D - 10D = r

  def __call__(self, x_in):
    return self.get_features(x_in)

def define_rand_feats(X, D=128):
  tf.random.set_seed(123129) # For reproducibility
  from scipy.spatial import distance
  rprm = np.random.permutation(X.shape[0])
  ds = distance.cdist(X[rprm[:100], :], X[rprm[100:], :])
  sigma_rot = np.mean(np.sort(ds)[:, 5])
  model = RandFeats(sigma_rot, X.shape[1], D)

  return model