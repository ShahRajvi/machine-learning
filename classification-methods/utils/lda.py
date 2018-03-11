import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import eig
from numpy.linalg import eigh

class LDA:
    ''' 
    @ Authors: Rajeev Bhatt Ambati, Zhongling Sun.
        ************* LINEAR DISCRIMINANT ANALYSIS ****************
        It has three functions train, test, decision_boundary used for training, testing and
        plotting the decision boundary respectively.
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
    def __init__(self, K):
    # Number of classes
        self.K = K
        self.params = {}
    
    def train(self, X_train, y_train):
    # Number of classes
        K = self.K
    # Dimensions of the training data
        N, p = X_train.shape
    # Prior probabilities
        P = []
        for k in range(K):
            Pk = np.mean(y_train == k)
            P.append(Pk)
    # Class-wise Mean
        mean_vectors = []
        for k in range(K):
            mu_k = np.mean(X_train[y_train == k], axis=0, keepdims=True)
            mean_vectors.append(mu_k)
    # Sample Variance
        sigma = np.zeros((p, p))
        for k, mu_k in zip(range(K), mean_vectors):
            sigma_k = np.zeros((p, p))
            for Xk in X_train[y_train == k]:
                sigma_k += np.dot((Xk - mu_k).T, (Xk - mu_k))
            sigma += sigma_k/(N - K)
    # Storing the parameters of the model
        self.params['Priors'] = P
        self.params['class_mean'] = mean_vectors
        self.params['covariance'] = sigma
        
        return
    
    def test(self, X_test, y_test):
    # No. of classes
        K = self.K
    # Dimensions of test data sample
        N, p = X_test.shape
    # Parameters
        P = self.params['Priors']
        mean_vectors = self.params['class_mean']
        sigma = self.params['covariance']
    # Classification
    # Rule: select a class 'k' which gives maximum delta
        inv_sigma = pinv(sigma)
        deltas = np.zeros((N, K))
        for k,mu_k,Pk in zip(range(K), mean_vectors, P):
            delta_k = (X_test.dot(inv_sigma.dot(mu_k.T)) -\
                       0.5*mu_k.dot(inv_sigma.dot(mu_k.T)) +\
                       np.log(Pk)
                      )
            deltas[:, k] = delta_k.reshape((N,))

        y_pred = np.argmax(deltas, axis=1)
        accuracy = np.mean(y_pred == y_test)*100.
        error = 100 - accuracy
        
        return y_pred, error
    
    def decision_boundary(self, X, y, xx_min, xx_max):
    # Considering only two directions for the plot
        P = self.params['Priors']
        P1, P2 = P[0], P[1]
        mean_vectors = self.params['class_mean']
        mu1, mu2 = mean_vectors[0], mean_vectors[1]
        sigma = self.params['covariance']
    # LDA Coefficients
        mid_point = (mu1 + mu2)/2 - np.mean(X, axis=0, keepdims=True)
        mid_point = mid_point[0, :2]

        orth_dir = (mu1 - mu2).dot(inv(sigma))[0, :2]

        m = orth_dir[0]/orth_dir[1]
        y1 = mid_point[1]
        x1 = mid_point[0]
    # Shifting the origin to the mean of the data
        X_w = X - np.mean(X, axis=0, keepdims=True)
        for label,marker,color in zip(range(3),('^', 'o'),('green', 'red')):
            plt.scatter(x=X_w[:,0].real[y == label],
                        y=X_w[:,1].real[y == label],
                        marker=marker,
                        color=color,
                        alpha=0.5)
        plt.legend(['benign', 'malignant'])
        plt.xlabel('dimension 1')
        plt.ylabel('dimension 2')
        plt.title('Decision boundary with LDA')
    # Decision boundary
        xx = np.linspace(xx_min, xx_max)
        # Equation of a straight line passing through (x1, y1) with slope 'm'
        yy = y1 + m*(xx - x1)
        plt.plot(xx, yy, 'k-')
        plt.grid(linestyle = 'dotted')