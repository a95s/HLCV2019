from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4, use_dropout=False, keep_prob=0.0):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.num_classes  = output_size
        self.use_dropout  = use_dropout
        self.keep_prob    = keep_prob

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = 0.0
        #############################################################################
        # TODO: Perform the forward pass, computing the class probabilities for the #
        # input. Store the result in the scores variable, which should be an array  #
        # of shape (N, C).                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # print("params. shapes")
        # print("fwd pass")
        # print("X ", X.shape)
        # print("W1 ", W1.shape)
        # print("W2 ", W2.shape)
        # print("b1 ", b1.shape)
        # print("b2 ", b2.shape)

        a1 = X
        z2 = np.matmul(X, W1) + b1

        a2 = np.maximum(0, z2)

        d2 = np.ones((a2.shape[0], a2.shape[1]))
        if self.use_dropout:
            d2 = np.random.rand(a2.shape[0], a2.shape[1])
            d2 = (d2 < self.keep_prob).astype(float)
            a2 = a2 * d2
            a2 = a2 / self.keep_prob

        z3 = np.matmul(a2, W2) + b2
        a3 = np.exp(z3) * np.expand_dims(1.0 / np.sum(np.exp(z3), axis=1), axis=1)
        scores = a3
        #print("intermediate layer shapes")
        #print(a1.shape, z2.shape, a2.shape, z3.shape, a3.shape)
        #print(a3.shape)
        #print(a3)
        #print(y)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.0
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Implement the loss for softmax output layer
        #loss = ((np.sum(-np.log(a3)))  + reg*(np.linalg.norm(W1,2) + np.linalg.norm(W2,2)))
        loss = -(np.sum(np.log(a3[range(N),y]))) / N + reg*(np.sum(W1*W1) + np.sum(W2*W2))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # delta matrix appears in formulas
        #print("calc. loss")
        # 3 classes: 0,1,2
        delta = np.zeros((N, self.num_classes))
        delta[np.arange(y.size), y] = 1         # to one-hot

        dZ3 = (a3 - delta)
        dW2 = np.matmul(a2.T, dZ3) / N + 2 * reg * W2
        db2 = np.sum(dZ3.T,axis=1) / N

        dA2 = np.matmul(dZ3, W2.T)

        if self.use_dropout:
            dA2 = dA2 * d2
            dA2 = dA2 / self.keep_prob

        #dZ2 = dA2
        dZ2 = np.multiply(dA2, np.int64(a2 > 0))
        dW1 = np.matmul(a1.T,dA2) / N + 2 * reg * W1
        db1 = np.sum(dA2.T, axis=1) / N

        #print("intermediate grad shapes")
        #print(dZ3.shape,dA2.shape,dZ2.shape,dW2.shape,db2.shape,dW1.shape,db1.shape)

        #dz3, dA2, dz2, db1

        #if self.use_dropout:
        #    dA2 = dA2 * d2
        #    dA2 = dA2 / self.keep_prob

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        # grads['W1'] = np.matmul(W2, np.matmul((a3 - delta).T, a1)).T / N + 2 * reg * W1
        # grads['b1'] = np.sum(np.matmul(W2, (a3 - delta).T), axis=1) / N
        # grads['W2'] = np.matmul(a2.T, (a3 - delta)) / N + 2 * reg * W2
        # grads['b2'] = np.sum((a3 - delta).T, axis=1) / N

        # print("Grads. shapes")
        # print("backwd pass")
        # print("dW1 ", grads['W1'].shape)
        # print("dW2 ", grads['W2'].shape)
        # print("db1 ", grads['b1'].shape)
        # print("db2 ", grads['b2'].shape)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            #X_batch = X
            #y_batch = y
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # get batch_size random entries, so we can get a random batch from set
            random_indices = np.random.randint(0, num_train, size=batch_size)
            #print(random_indices)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
            self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']
            self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
            self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']
            # print("after update")
            # print("W1 ", self.params['W1'].shape)
            # print("W2 ", self.params['W2'].shape)
            # print("b1 ", self.params['b1'].shape)
            # print("b2 ", self.params['b2'].shape)
            #exit(0)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        y_pred = np.argmax(self.loss(X), axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
