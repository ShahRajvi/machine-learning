'''
*********************** MAIN FILE *****************************
@ Authors: Rajeev Bhatt Ambati and Zhongling Sun.
    1) This is the main file for project 1 of STAT 557: Data Mining I course.
    2) 3 Classification methods Linear Discriminant Analysis (LDA),
       Quadratic Discriminant Analysis (QDA), Logistic Regression are explored.
    3) Dimension Reduction methods LDA, Principal Component Analysis (PCA) are also applied.
    4) All the results are performed on Breast Cancer Wisconsin (Diagnostic) Data Set
'''

########### Importing Libraries ###########
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

########### Supporting Classes ############
from utils.lda import LDA
from utils.qda import QDA
from utils.binary_logistic import binary_logistic
from utils.reduce_LDA import reduce_LDA
from utils.reduce_PCA import reduce_PCA

# Directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')
np.seterr(divide='ignore')

########### Used Train and Test Data ########
data = io.loadmat('used_data.mat')
X_train, y_train = data['train_data'], data['train_labels'].reshape((X_train.shape[0],))
X_test, y_test = data['test_data'], data['test_labels'].reshape((X_test.shape[0],))

print 'Training data size =', X_train.shape
print 'Training labels size =', y_train.shape
print 'Testing data size =', X_test.shape
print 'Testing labels size =', y_test.shape

####### Linear Discriminant Analysis ######
bc_lda = LDA(K = 2)

bc_lda.train(X_train, y_train)
pred, error = bc_lda.test(X_test, y_test)

print '######### Applying Linear Discriminant Analysis ########'
print 'Testing error obatined =', error

bc_lda.decision_boundary(X_test, y_test, xx_min = 0.3, xx_max = 1)
plt.savefig('plots/p1_LDA_decision_boundary.jpg', dpi=300)
plt.show()

#### Quadratic Discriminant Analysis #####
bc_qda = QDA(K = 2)

bc_qda.train(X_train, y_train)
pred, error = bc_qda.test(X_test, y_test)

print '###### Applying Quadratic Discriminant Analysis ########'
print 'Testing error obtained =', error

bc_qda.decision_boundary(X_test, y_test, xx_min = 0.1, xx_max = 7)
plt.savefig('plots/p1_QDA_decision_boundary.jpg', dpi=300)
plt.show()

########## Logistic Regression ###########
bin_logit = binary_logistic()

bin_logit.train(X_train, y_train, iters=6)
pred, error = bin_logit.test(X_test, y_test)

print '################# Logistic Regression ##################'
print 'Testing error obtained =', error

bin_logit.training_loss_plot()
plt.savefig('plots/p1_logit_lossVsiter.jpg', dpi=300)
plt.show()

bin_logit.decision_boundary(X_test, y_test, xx_min = -0.5, xx_max = +0.5)
plt.savefig('plots/p1_logit_decision_boundary.jpg', dpi=300)
plt.show()

#################### Dimension reduction with LDA ####################
print '####### Reducing the dimension of data with LDA ########'
red_lda = reduce_LDA(K = 2)
V,_ = red_lda.subspace(X_train[:, :2], y_train)

red_lda.lda_axis(X_test, y_test, xx1_min = -5, xx1_max = +5, xx2_min = -2, xx2_max = +2)
plt.savefig('plots/p1_LDA_coordinates.jpg', dpi=300)
plt.show()

red_lda.projection(X_test, y_test, yy1_min = -10, yy1_max = 10, xx2_min = -5, xx2_max = 5)
plt.savefig('plots/p1_vis_data_reduced_with_LDA.jpg', dpi=300)
plt.show()

# Applying LDA on the reduced data
red_lda = reduce_LDA(K = 2)
V, D = red_lda.subspace(X_train, y_train)

# Variance spread bar graph
plt.bar(range(1,31), 100*abs(D)/sum(abs(D)), color='green')
plt.title('Variance captured by each eigen vector')
plt.xlabel('Discriminant Coordinates')
plt.ylabel('Variance (%)')
plt.grid(linestyle='dotted')
plt.savefig('plots/p1_LDA_variance_spread.jpg', dpi=300)
plt.show()

train_error_mat = []
test_error_mat = []
min_test_error = 100
for i in range(1, 30):
    X_train_red = X_train.dot(V[:, :i])
    X_test_red = X_test.dot(V[:, :i])

    bc_red_lda = LDA(K=2)
    bc_red_lda.train(X_train_red, y_train)
    train_pred, train_error = bc_red_lda.test(X_train_red, y_train)
    test_pred, test_error = bc_red_lda.test(X_test_red, y_test)
    
    train_error_mat.append(train_error)
    test_error_mat.append(test_error)
    
    if test_error < min_test_error:
        min_test_error = test_error
        good_dims = i

plt.plot(range(1, 30), train_error_mat, 
         range(1, 30), test_error_mat)
plt.xlabel('No. of dimensions')
plt.ylabel('Classification Error')
plt.legend(['Training', 'Testing'])
plt.grid(linestyle = 'dotted')
s = 'minimum test error = ' + r'%.2f%%' % min_test_error + ' for d = ' + str(good_dims)
plt.title(s)
plt.savefig('plots/p1_LDA_on_data_reduced_with_LDA.jpg', dpi=300)
plt.show()

print 'Applying LDA on this data gives minimum test error as %.2f \
for %d dimensions'% (min_test_error, good_dims)

# Applying QDA on the reduced data
train_error_mat = []
test_error_mat = []
min_test_error = 100
for i in range(1, 30):
    X_train_red = X_train.dot(V[:, :i])
    X_test_red = X_test.dot(V[:, :i])

    bc_red_qda = QDA(K=2)
    bc_red_qda.train(X_train_red, y_train)
    train_pred, train_error = bc_red_qda.test(X_train_red, y_train)
    test_pred, test_error = bc_red_qda.test(X_test_red, y_test)
    
    train_error_mat.append(train_error)
    test_error_mat.append(test_error)
    
    if test_error < min_test_error:
        min_test_error = test_error
        good_dims = i

plt.plot(range(1, 30), train_error_mat, 
         range(1, 30), test_error_mat)
plt.xlabel('No. of dimensions')
plt.ylabel('Classification Error')
plt.legend(['Training', 'Testing'])
plt.grid(linestyle = 'dotted')
s = 'minimum test error = ' + r'%.2f%%' % min_test_error + ' for d = ' + str(good_dims)
plt.title(s)
plt.savefig('plots/p1_QDA_on_data_reduced_with_LDA.jpg', dpi=300)
plt.show()

print 'Applying QDA on this data gives minimum test error as %.2f for \
%d dimensions'% (min_test_error, good_dims)

# Applying Logistic Regression on the reduced data
train_error_mat = []
test_error_mat = []
min_test_error = 100
for i in range(1, 30):
    X_train_red = X_train.dot(V[:, :i])
    X_test_red = X_test.dot(V[:, :i])

    bc_red_logit = binary_logistic()
    bc_red_logit.train(X_train_red, y_train, iters=6)
    train_pred, train_error = bc_red_logit.test(X_train_red, y_train)
    test_pred, test_error = bc_red_logit.test(X_test_red, y_test)
    
    train_error_mat.append(train_error)
    test_error_mat.append(test_error)
    
    if test_error < min_test_error:
        min_test_error = test_error
        good_dims = i

plt.plot(range(1, 30), train_error_mat, 
         range(1, 30), test_error_mat)
plt.xlabel('No. of dimensions')
plt.ylabel('Classification Error')
plt.legend(['Training', 'Testing'])
plt.grid(linestyle = 'dotted')
s = 'minimum test error = ' + r'%.2f%%' % min_test_error + ' for d = ' + str(good_dims)
plt.title(s)
plt.savefig('plots/p1_logit_on_data_reduced_with_LDA.jpg', dpi=300)
plt.show()

print 'Applying logistic regression on this data gives minimum test error \
as %.2f for %d dimensions'% (min_test_error, good_dims)

########### Dimension Reduction with PCA ##########
print '####### Reducing the dimension of data with PCA ########'
########### Dimension Reduction with PCA ##########
red_pca = reduce_PCA(K = 2)
V,_ = red_pca.subspace(X_train[:, :2], y_train)

red_pca.pca_axis(X_test, y_test, xx1_min = -3, xx1_max = +3, xx2_min = -3, xx2_max = +3)
plt.savefig('plots/p1_PCA_components.jpg', dpi=300)
plt.show()
red_pca.projection(X_test, y_test, yy1_min = -7, yy1_max = 7, xx2_min = -7, xx2_max = 7)
plt.savefig('plots/p1_vis_data_reduced_with_PCA.jpg', dpi=300)
plt.show()

red_pca = reduce_PCA(K = 2)
V, D = red_pca.subspace(X_train, y_train)

# Variance spread bar graph
plt.bar(range(1,31), 100*abs(D)/sum(abs(D)), color='green')
plt.title('Variance captured by each eigen vector')
plt.xlabel('Principal Component Directions')
plt.ylabel('Variance (%)')
plt.grid(linestyle='dotted')
plt.savefig('plots/p1_PCA_variance_spread.jpg', dpi=300)
plt.show()

# Applying LDA on the reduced data
train_error_mat = []
test_error_mat = []
min_test_error = 100
for i in range(1, 30):
    X_train_red = X_train.dot(V[:, :i])
    X_test_red = X_test.dot(V[:, :i])

    bc_red_lda = LDA(K=2)
    bc_red_lda.train(X_train_red, y_train)
    train_pred, train_error = bc_red_lda.test(X_train_red, y_train)
    test_pred, test_error = bc_red_lda.test(X_test_red, y_test)
    
    train_error_mat.append(train_error)
    test_error_mat.append(test_error)
    
    if test_error < min_test_error:
        min_test_error = test_error
        good_dims = i

plt.plot(range(1, 30), train_error_mat, 
         range(1, 30), test_error_mat)
plt.xlabel('No. of dimensions')
plt.ylabel('Classification Error')
plt.legend(['Training', 'Testing'])
plt.grid(linestyle = 'dotted')
s = 'minimum test error = ' + r'%.2f%%' % min_test_error + ' for d = ' + str(good_dims)
plt.title(s)
plt.savefig('plots/p1_LDA_on_data_reduced_with_PCA.jpg', dpi=300)
plt.show()

print 'Applying LDA on this data gives minimum test error as %.2f \
for %d dimensions'% (min_test_error, good_dims)

# Applying QDA on the reduced data
train_error_mat = []
test_error_mat = []
min_test_error = 100
for i in range(1, 30):
    X_train_red = X_train.dot(V[:, :i])
    X_test_red = X_test.dot(V[:, :i])

    bc_red_qda = QDA(K=2)
    bc_red_qda.train(X_train_red, y_train)
    train_pred, train_error = bc_red_qda.test(X_train_red, y_train)
    test_pred, test_error = bc_red_qda.test(X_test_red, y_test)
    
    train_error_mat.append(train_error)
    test_error_mat.append(test_error)
    
    if test_error < min_test_error:
        min_test_error = test_error
        good_dims = i

plt.plot(range(1, 30), train_error_mat, 
         range(1, 30), test_error_mat)
plt.xlabel('No. of dimensions')
plt.ylabel('Classification Error')
plt.legend(['Training', 'Testing'])
plt.grid(linestyle = 'dotted')
s = 'minimum test error = ' + r'%.2f%%' % min_test_error + ' for d = ' + str(good_dims)
plt.title(s)
plt.savefig('plots/p1_QDA_on_data_reduced_with_PCA.jpg', dpi=300)
plt.show()

print 'Applying QDA on this data gives minimum test error as %.2f \
for %d dimensions'% (min_test_error, good_dims)

# Applying Logistic Regression on the reduced data

train_error_mat = []
test_error_mat = []
min_test_error = 100
for i in range(1, 30):
    X_train_red = X_train.dot(V[:, :i])
    X_test_red = X_test.dot(V[:, :i])

    bc_red_logit = binary_logistic()
    bc_red_logit.train(X_train_red, y_train, iters=5)
    train_pred, train_error = bc_red_logit.test(X_train_red, y_train)
    test_pred, test_error = bc_red_logit.test(X_test_red, y_test)
    
    train_error_mat.append(train_error)
    test_error_mat.append(test_error)
    
    if test_error < min_test_error:
        min_test_error = test_error
        good_dims = i

plt.plot(range(1, 30), train_error_mat, 
         range(1, 30), test_error_mat)
plt.xlabel('No. of dimensions')
plt.ylabel('Classification Error')
plt.legend(['Training', 'Testing'])
plt.grid(linestyle = 'dotted')
s = 'minimum test error = ' + r'%.2f%%' % min_test_error + ' for d = ' + str(good_dims)
plt.title(s)
plt.savefig('plots/p1_logit_on_data_reduced_with_PCA.jpg', dpi=300)
plt.show()

print 'Applying logistic regression on this data gives minimum test error \
as %.2f for %d dimensions'% (min_test_error, good_dims)