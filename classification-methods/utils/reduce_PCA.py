import numpy as np
from numpy.linalg import eig
from numpy.linalg import eigh
from sklearn.preprocessing import StandardScaler

class reduce_PCA:
    '''
    @ Authors: Rajeev Bhatt Ambati, Zhongling Sun.
        ********************* DIMENSION REDUCTION USING PCA *******************
        This class has three functions subspace, pca_axis, projection used for extracting
        PCA subspace 'V', plotting the subspace, projecting the given data respectively.
        Args:
            Inputs: X_train -- training data matrix.
                    y_train -- training labels.
                    X_test  -- testing data matrix.
                    y_test  -- testing labels.
                    K       -- No. of classes.
                    xx1_min, xx1_max -- Plotting range for the first principal component.
                    xx2_min, xx2_max -- Plotting range for the second principal component.
            Outputs: y_pred -- Labels classified by the trained LDA model, size [N x 1].
                     Vs     -- LDA vector subspace 
                     Ds     -- eigen vectors corresponding to the linear discriminants.
    '''
    def __init__(self, K):
        self.K = K
        self.params = {}
    
    def subspace(self, X_train, y_train):
    # Number of classes
        K = self.K
        X_std = StandardScaler().fit_transform(X_train)
    # Dimensions of data
        N, p = X_std.shape
    # Mean and variance of the data
        mu = np.mean(X_std, axis=0, keepdims=True)
        sigma = ((X_std - mu).T).dot(X_std - mu)/(N - 1)
    # Eigen decomposition of covariance matrix
        D, V = eigh(sigma)
    # Sorting the eigen values and the corresponding eigen vectors
        indcs = [i[0] for i in sorted(enumerate(abs(D)), key=lambda x: x[1], reverse=True)]
        Ds = D[indcs]
        Vs = V[:, indcs]
    # Storing the eigen vectors
        self.params['V'] = Vs
        self.params['D'] = Ds
        return Vs, Ds
    
    def pca_axis(self, X_test, y_test, xx1_min, xx1_max, xx2_min, xx2_max):
    # Number of classes
        K = self.K
    # Parameters
        V = self.params['V']
        
        X_test_w = X_test - np.mean(X_test, axis=0, keepdims=True)
        for label,marker,color in zip(range(1+K),('^', 'o'),('green', 'red')):
            plt.scatter(x=X_test_w[:,0].real[y_test == label],
                        y=X_test_w[:,1].real[y_test == label],
                        marker=marker,
                        color=color,
                        alpha=0.5)
        plt.xlabel('dimension 1')
        plt.ylabel('dimension 2')
        plt.title('Principal Component Directions')
    # PCA axis
        xx1 = np.linspace(xx1_min, xx1_max)
        xx2 = np.linspace(xx2_min, xx2_max)
    # PC 1
        yy1 = (V[1,0]/V[0,0])*xx1
    # PC 2
        yy2 = (V[1,1]/V[0,1])*xx2

        plt.plot(xx1, yy1, 'k') 
        plt.plot(xx2, yy2, 'k--')
        plt.ylim(-10, 7)
        plt.xlim(-7, 10)
        plt.legend(['PC 1', 'PC 2', 'benign', 'malignant'])
        plt.grid(linestyle = 'dotted')
        
    def projection(self, X, y, yy1_min, yy1_max, xx2_min, xx2_max):
    # Number of classes
        K = self.K
    # Parameters
        V = self.params['V']
    # Projection
        X_red = X[:, :2].dot(V)
        
        X_red_w = X_red - np.mean(X_red, axis=0, keepdims=True)
        for label,marker,color in zip(range(1+K),('^', 'o'),('green', 'red')):
            plt.scatter(x=X_red_w[:,0].real[y_test == label],
                        y=X_red_w[:,1].real[y_test == label],
                        marker=marker,
                        color=color,
                        alpha=0.5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA: Projection on to first 2 principal components')
        plt.legend(['benign', 'malignant'])

    # PC 1
        yy1 = np.linspace(yy1_min, yy1_max)
        xx1 = (0)*yy1
    # PC 2
        xx2 = np.linspace(xx2_min, xx2_max)
        yy2 = (0)*xx2
        
        plt.plot(xx1, yy1, 'k')
        plt.plot(xx2, yy2, 'k--')
        plt.grid(linestyle = 'dotted')