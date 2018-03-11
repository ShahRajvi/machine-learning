import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import eig
from numpy.linalg import eigh

class QDA:
    ''' 
    @ Authors: Rajeev Bhatt Ambati, Zhongling Sun.
        ************* QUADRATIC DISCRIMINANT ANALYSIS ****************
        QDA Class has three functions train, test, decision_boundary used for training, testing 
        and plotting the decision boundary respectively.
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
    # Within Class-Variance
        sigmas = []
        for k, mu_k in zip(range(K), mean_vectors):
            sigma_k = np.zeros((p, p))
            for Xk in X_train[y_train == k]:
                sigma_k += ((Xk - mu_k).T).dot(Xk - mu_k)
            sigma_k = sigma_k/np.sum(y_train == k)
            sigmas.append(sigma_k)
    # Storing the parameters
        self.params['Priors'] = P
        self.params['class_mean'] = mean_vectors
        self.params['class_variance'] = sigmas
        
        return
    def test(self, X_test, y_test):
    # Number of classes
        K = self.K
    # Dimensions of test data sample
        N, p = X_test.shape
    # Parameters
        P = self.params['Priors']
        mean_vectors = self.params['class_mean']
        sigmas = self.params['class_variance']
    # Classification
    # Select a class 'k' which gives maximum delta
        deltas = np.zeros((N, K))
        for k,sigma_k,mu_k,Pk in zip(range(K), sigmas, mean_vectors, P):
            inv_sigma_k = inv(sigma_k)
            delta_k = np.diag(-0.5*np.log(det(sigma_k))\
                              -0.5*(X_test - mu_k).dot(inv_sigma_k.dot((X_test - mu_k).T))\
                              +np.log(Pk)
                             )
            deltas[:, k] = delta_k

        y_pred = np.argmax(deltas, axis=1)
        accuracy = np.mean(y_pred == y_test)*100.
        error = 100 - accuracy
        
        return y_pred, error
    
    def decision_boundary(self,X, y, xx_max, xx_min):
        P = self.params['Priors']
        P1, P2 = P[0], P[1]
        mean_vectors = self.params['class_mean']
        mu1, mu2 = mean_vectors[0][:, :2], mean_vectors[1][:, :2]
        sigmas = self.params['class_variance']
        sigma1, sigma2 = sigmas[0][:2, :2], sigmas[1][:2, :2]
    # QDA coefficients
        p = inv(sigma2) - inv(sigma1)
        q = inv(sigma2).dot(mu2.T) - inv(sigma1).dot(mu1.T) + ((mu2.dot(inv(sigma2)) - mu1.dot(inv(sigma1))).T)
        r = (mu2.dot(inv(sigma2).dot(mu2.T)) - mu1.dot(inv(sigma1).dot(mu1.T)))[0] + \
            np.log(det(sigma2)/det(sigma1)) - 2*np.log(P2/P1)

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
        plt.title('Decision boundary with QDA')
    # Decision boundary
        xx = np.linspace(xx_min, xx_max)
        a = p[1,1]
        b = xx*(p[0,1] + p[1,0] - q[1,0])
        c = r - (xx**2)*p[0,0] - xx*q[0, 0]
        D = (b**2) - 4*a*c

        yy1 = (-b + np.sqrt(D))/(2*a)
        #yy2 = (-b - np.sqrt(D))/(2*a)

        plt.plot(xx, yy1, 'k')
        #plt.plot(xx, yy2, 'k--')
        plt.grid(linestyle = 'dotted')