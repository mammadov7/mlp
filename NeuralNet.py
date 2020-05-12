import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import utility as ut

class NeuralNet:
  """
    — un nombre de couches cachées et leur dimensions
    — une liste de matrices de poids
    — une liste de matrices de biais
    — une liste de matrices d’entrées pondérées
    — une liste de matrices d’activations
    — un taux d’apprentissage (η)
    — une fonction d’activation pour les unités des couches cachées
    — un nombre d’epoch pendant lequel entrainer le modéle
  """
  bias = []
  weights = []
  X_train, X_test = [], []
  y_train, y_test = [], []
  hidden_layer_sizes = []
  layer_sizes = []
  A, df = [],[]
  activation = ''
  learning_rate = 0.01
  epoch = 200

  def __init__(self, X_train = None, y_train = None, X_test = None, y_test = None, \
                hidden_layer_sizes = [4,3,2] , activation='identity', learning_rate=0.01, epoch=200):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.learning_rate = learning_rate
    self.epoch = epoch
    self.layer_sizes =  [len ( X_train.columns )] + hidden_layer_sizes + [len( y_train.columns )] 
    self._weights_initialization()

  def _weights_initialization(self):
    self.A = [None] * ( len(self.hidden_layer_sizes) + 1 )
    self.df = [None] * ( len(self.hidden_layer_sizes) + 1 )
    for i in range (1,len(self.layer_sizes) ):
      # initialization of weights with Xavier method
      self.weights.append( np.array( self.layer_sizes[i]*[ self.layer_sizes[i-1]*[None] ] )  )
      self.weights[i-1] = np.random.normal(loc=0.0,scale=1.0 / np.sqrt(self.layer_sizes[i]),
                                            size=(self.layer_sizes[i], self.layer_sizes[i-1]))
      # initialization biases: just zeros
      self.bias.append( np.array( self.layer_sizes[i]*[None] ) )
      self.bias[i-1] = np.zeros( self.layer_sizes[i] )

  def feed_forward(self, X ,y):
    '''
    Implementation of the Feedforward
    '''
    # Fonction d'activation 
    g = lambda x: ut.tanh(x)
    Z = [None] * len(self.layer_sizes)
    input_layer = X
    for i in range(len(self.hidden_layer_sizes) + 1):
      # Multiplying input_layer by weights for this layer
      Z[i + 1] = np.dot(self.weights[i],input_layer) + self.bias[i]
      # Activation Function
      if( i == len(self.hidden_layer_sizes) ):
        # Just for output layer softmax()
        # For calculating the loss
        self.A[i] = ut.softmax(Z[i + 1])
        # Derivative of softmax, returns a matrix
        self.df[i] = ut.softmax_gradient( Z[i + 1] )
      else:
        # for the other layers tanh()
        self.A[i],self.df[i] = g(Z[i + 1])
      # Current output_layer will be next input_layer
      input_layer = self.A[i]
    error = ut.cross_entropy_loss(self.A[-1],y)
    return error , self.A[-1]

  def back_propagation(self, X, y):
    # Initialization
    delta = [None] * (len(self.hidden_layer_sizes) + 1)
    dW = [None] * (len(self.hidden_layer_sizes) + 1)
    db = [None] * (len(self.hidden_layer_sizes) + 1)
    # Calculation for last(output) layer
    # dot product of error and matrix( derivative of softmax )
    delta[-1] = np.dot((self.A[-1] - y),self.df[-1])
    dW[-1] = np.transpose(delta[-1] * ut.tr(self.A[-2]))
    db[-1] = delta[-1]
    # Calculation for the rest
    for l in range(len(self.hidden_layer_sizes) -1 , -1, -1):
      delta[l] = np.multiply(np.dot(self.weights[l + 1].T, delta[l + 1]), self.df[l])
      if( l == 0 ):
        dW[l] = np.transpose(delta[l] * ut.tr(X))
      else:
        dW[l] = np.transpose(delta[l] * ut.tr(self.A[l-1]))
      db[l] = delta[l]
    # Updating the Weights and Biases for next epoch
    for l in range(len(self.hidden_layer_sizes) + 1):
      self.weights[l] = self.weights[l] - self.learning_rate*dW[l]
      self.bias[l] = self.bias[l] - self.learning_rate*db[l]
        
  def predict(self, X , y):
    accuracies = []
    prediction = []
    for _X, _y in zip(X.values, y.values):
      err , A = self.feed_forward(_X,_y)
      prediction.append(A)
      accuracies.append( self.accuracy( A, _y ) )
    print("Accuracy: "+ str( np.mean(accuracies) ))
    return accuracies,prediction

  def train (self, X_train,y_train,X_test,y_test):
    mean_error_train, mean_error_test  = [], []
    for i in range(200):
  # One epoch
      error_train, error_test = [], []
      # Training 
      for _X, _y in zip(X_train.values, y_train.values):
        err, A = self.feed_forward(_X, _y)
        error_train.append(err)
        self.back_propagation(_X,_y)
      # Testing
      for _X, _y in zip(X_test.values, y_test.values):
        err, A = self.feed_forward(_X, _y)
        error_test.append(err)
      # Mean Error Cross-Entropy
      mean_error_train.append( np.mean( error_train ) )
      mean_error_test.append( np.mean( error_test ) )
      # Shuffle
      X_train, y_train = shuffle(X_train, y_train)
      X_test, y_test = shuffle(X_test, y_test)

    return mean_error_train, mean_error_test

  def accuracy( self, y_pred,y_real ):
    return int( ut.max_row(y_pred) == ut.max_row(y_real) )

  def confusion_matrix(self, y_pred, y_real):
    for i in range(len(y_pred)):
      y_pred[i] = ut.convert(y_pred[i])
    n = y_real.shape[1]
    m = np.array( n*[ n*[0] ] )
    for i in range(len(y_real)):
      if( y_real.values[i][0] == 1 ):
        m[0] += y_pred[i]
      elif(y_real.values[i][1] == 1 ):
        m[1] += y_pred[i]
      else:
        m[2] += y_pred[i]
    ut.plt_confusion_matrix(m)
    return m
