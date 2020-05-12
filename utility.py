import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def tanh(Z):
   """
   Z : non activated outputs
   Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
   """
   A = np.empty(Z.shape)
   A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
   df = 1-A**2
   return A,df

def softmax(z):
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)

def softmax_gradient(z):
    """Computes the gradient of the softmax function."""
    Sz = softmax(z)
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def cross_entropy_loss(p, y):
    """Cross-entropy loss between predicted and expected probabilities."""
    assert(p.shape == y.shape)
    return -np.sum(y * np.log(p))

def tr(x):
  """              [ [1],
      [ 1, 2, 4] ->  [2],
                     [4] ] 
  """
  x = np.array(  [[i] for i in x]  )
  return x

def max_row(row):
  """
    row: 1D array
    Returns: index of the maximum value in row
  """
  return np.where(row == np.amax(row))[0][0]
 
def convert(y_hat):
  """ Converting the predicted data: [0.23 0.2 0.85] -> [0 0 1] """
  y_max = np.max(y_hat)
  y = []
  for i in range(len(y_hat)):
    if y_hat[i] == y_max:
      y.append(1)
    else:
      y.append(0)
  return y

def plt_confusion_matrix( confusion_mtx = [], class_names = ['class-0', 'class-1', 'class-2'] ):
  plt.figure(figsize = (8,8))
  sns.set(font_scale=2) # label size
  ax = sns.heatmap(
      confusion_mtx, annot=True, annot_kws={"size": 30}, # font size
      cbar=False, cmap='Blues', fmt='d', 
      xticklabels=class_names, yticklabels=class_names)
  ax.set(title='', xlabel='Actual', ylabel='Predicted')
  plt.show()

def max_row(row):
  """
    row: 1D array
    Returns: index of the maximum value in row
  """
  return np.where(row == np.amax(row))[0][0] 