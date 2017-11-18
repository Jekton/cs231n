import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_data = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_data):
    x = X[i]
    f = np.matmul(x, W)
    f -= np.max(f)
    total = np.sum(np.exp(f))
    p = np.exp(f[y[i]]) / total
    loss += -np.log(p)
    for c in range(num_class):
      dW[:, c] += x * np.exp(f[c]) / total
    dW[:, y[i]] -= x
  loss /= num_data
  dW /= num_data
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_data = X.shape[0]
  F = np.matmul(X, W)
  F -= np.max(F, axis=1).reshape((-1, 1))
  exp_F = np.exp(F)
  sum_exp_F = np.sum(exp_F, axis=1)
  target_scores = F[range(num_data), y]
  loss = np.sum(np.log(sum_exp_F))
  loss -= np.sum(target_scores)
  loss /= num_data

  Frac = exp_F / sum_exp_F.reshape((-1, 1))
  Frac[range(num_data), y] -= 1
  dW = np.matmul(X.T, Frac)
  dW /= num_data
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

