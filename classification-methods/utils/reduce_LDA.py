import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eigh
from numpy.linalg import eig

class reduce_LDA:
    '''
    @ Authors: Rajeev Bhatt Ambati, Zhongling Sun.
        ********************* DIMENSION REDUCTION USING LDA *******************
        This class has three functions subspace, lda_axis, projection used for extracting
        LDA subspace 'V', plotting the subspace, projecting the given data respectively.
        Args:
            Inputs: X_train -- training data matrix.
                    y_train -- training labels.
                    X_test  -- testing data matrix.
                    y_test  -- testing labels.
                    K       -- No. of classes.
                    xx1_min, xx1_max -- Plotting range for the first linear discriminant.
                    xx2_min, xx2_max -- Plotting range for the second linear discriminant.
            Outputs: y_pred -- Labels classified by the trained LDA model, size [N x 1].
                     Vs     -- LDA vector subspace 
                     Ds     -- eigen vectors corresponding to the linear discriminants.
    '''    
    def __init__(self, K):
        self.K = K
        self.params = {}
    def subspace(self, X_train, y_train):
    # Number of Classes
        K = self.K
    # Dimensions of the data
        N, p = X_train.shape
    # Prior probabilities
        P = []
        for k in range(K):
            Pk = np.mean(y_train == k)
            P.append(Pk)
    # Overall mean
        mu = np.mean(X_train, axis=0, keepdims=True)
    # Class-wise mean
        mean_vectors = []
        for k in range(K):
            mu_k = np.mean(X_train[y_train == k], axis=0, keepdims=True)
            mean_vectors.append(mu_k)
    # Within-class covariance matrix
        Sw = np.zeros((p, p))
        for k, mu_k in zip(range(K), mean_vectors):
            covar_k = np.zeros((p, p))            # Covariance of class 'k'            
            for Xk in X_train[y_train == k]:
                covar_k += ((Xk - mu_k).T).dot(Xk - mu_k)
            Sw += covar_k
    # Between-class covariance matrix
        Sb = np.zeros((p, p))
        for k, mu_k, Pk in zip(range(K), mean_vectors, P):
            Sb += N*Pk*((mu_k - mu).T).dot(mu_k - mu)
    # Eigen decomposition of inv(Sw)Sb
        D, V = eigh(pinv(Sw).dot(Sb))
    # Sorting the eigen values and the corresponding eigen vectors
        indcs = [i[0] for i in sorted(enumerate(abs(D)), key=lambda x: x[1], reverse=True)]
        Ds = D[indcs]
        Vs = V[:, indcs]
    # Storing the eigen vectors
        self.params['V'] = Vs
        self.params['D'] = Ds
        return Vs, Ds

    def lda_axis(self, X_test, y_test, xx1_min, xx1_max, xx2_min, xx2_max):
    # Parameters
        V = self.params['V']
        
        X_test_w = X_test - np.mean(X_test, axis=0, keepdims=True)
        for label,marker,color in zip(range(3),('^', 'o'),('green', 'red')):
            plt.scatter(x=X_test_w[:,0].real[y_test == label],
                        y=X_test_w[:,1].real[y_test == label],
                        marker=marker,
                        color=color,
                        alpha=0.5)
        plt.xlabel('dimension 1')
        plt.ylabel('dimension 2')
        plt.title('Linear Discriminant Coordinates')
    # LDA axis
        xx1 = np.linspace(xx1_min, xx1_max)
        xx2 = np.linspace(xx2_min, xx2_max)
        # LD 1
        yy1 = (V[1,0]/V[0,0])*xx1
        # LD 2
        yy2 = (V[1,1]/V[0,1])*xx2

        plt.plot(xx1, yy1, 'k') 
        plt.plot(xx2, yy2, 'k--')
        plt.ylim(-10, 7)
        plt.legend(['LD 1', 'LD 2', 'benign', 'malignant'])
        plt.grid(linestyle = 'dotted')
        
    def projection(self, X, y, yy1_min, yy1_max, xx2_min, xx2_max):
    # Parameters
        V = self.params['V']
    # Projection
        X_red = X[:, :2].dot(V)
        
        X_red_w = X_red - np.mean(X_red, axis=0, keepdims=True)
        for label,marker,color in zip(range(3),('^', 'o'),('green', 'red')):
            plt.scatter(x=X_red_w[:,0].real[y_test == label],
                        y=X_red_w[:,1].real[y_test == label],
                        marker=marker,
                        color=color,
                        alpha=0.5)
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.title('LDA: Projection on to first 2 discriminant coordinates')
        plt.legend(['benign', 'malignant'])

    # LD 1
        yy1 = np.linspace(yy1_min, yy1_max)
        xx1 = (0)*yy1
    # LD 2
        xx2 = np.linspace(xx2_min, xx2_max)
        yy2 = (0)*xx2
        
        plt.plot(xx1, yy1, 'k')
        plt.plot(xx2, yy2, 'k--')
        plt.grid(linestyle = 'dotted')