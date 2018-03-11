import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import eig
from numpy.linalg import eigh

class binary_logistic:
    ''' 
    @ Authors: Rajeev Bhatt Ambati, Zhongling Sun.
        ************* BINARY LOGISTIC REGRESSION ****************
        It has three functions train, test, training_loss_plot, decision_boundary used for 
        training, testing, plotting the training loss versus epochs and plotting 
        the decision boundary respectively.
        Args:
            Inputs: X_train -- training data matrix.
                    y_train -- training labels.
                    X_test  -- testing data matrix.
                    y_test  -- testing labels.
                    K       -- No. of classes.
                    xx_min, xx_max -- Range in which the decision boundary is plotted.
            Outputs: y_pred -- Labels classified by the trained LDA model, size [N x 1].
                     error  -- Classification error.
    '''
    def __init__(self):
        self.params = {}
    
    def train(self, X_train, y_train, iters):
        self.iters = iters
    # Dimensions of the data
        N, p = X_train.shape
    # Initializing beta
        beta = np.zeros((p, ))
        losses = []
        for i in range(iters):
            p_x = np.exp(X_train.dot(beta))/(1 + np.exp(X_train.dot(beta)))
            p_x = p_x.reshape((N,))
        # Log likelihood loss
            loss = np.mean(-y_train*np.log(p_x) - (1 - y_train)*np.log(1 - p_x))
            losses.append(loss)
            W = np.diag(p_x*(1-p_x))
            z = X_train.dot(beta) + pinv(W).dot(y_train - p_x)
            beta = pinv((X_train.T).dot(W.dot(X_train))).dot((X_train.T).dot(W.dot(z)))
    # Saving the losses
        self.params['beta'] = beta
        self.params['training_costs'] = losses
        
        return
    
    def test(self, X_test, y_test):
    # Parameters
        beta = self.params['beta']
    # Classification
    # Rule: y_pred = 1 if beta'X>=0 and 0 if beta'X<0
        deltas = X_test.dot(beta)
        y_pred = np.array(deltas>=0, dtype=int)
        accuracy = np.mean(y_pred == y_test)*100.
        error = 100 - accuracy
        
        return y_pred, error
    
    def training_loss_plot(self):
        iters = self.iters
        losses = self.params['training_costs']
    # Plot os loss function versus iterations
        plt.plot(range(iters), losses)
        plt.xlabel('No. of iterations')
        plt.ylabel('loss')
        plt.title('loss vs iterations')

    def decision_boundary(self, X, y, xx_min, xx_max):
    # Logistic Regression coefficients
        beta = self.params['beta']

        X_w = X - np.mean(X, axis=0, keepdims=True)

        for label,marker,color in zip(range(3),('^', 'o'),('green', 'red')):
            plt.scatter(x=X_w[:,0].real[y_test == label],
                        y=X_w[:,1].real[y_test == label],
                        marker=marker,
                        color=color,
                        alpha=0.5)
        plt.legend(['benign', 'malignant'])
        plt.xlabel('dimension 1')
        plt.ylabel('dimension 2')
        plt.title('Decision boundary with Logistic Regression')
    # Decision Boundary
        xx = np.linspace(xx_min, xx_max)
        yy = -(beta[0]/beta[1])*xx

        plt.plot(xx, yy, 'k-')
        plt.grid(linestyle = 'dotted')