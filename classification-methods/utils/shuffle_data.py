'''
***************************** DATASET ********************************
@ Authors: Rajeev Bhatt Ambati, Zhongling Sun.
    1) This file is used to shuffle the data using python function random.shuffle.
    2) The first 70% of the shuffled data is used as the Training data.
    3) The last 30% of the shuffled data is used as the Testing data.
    4) Finally its saved as a .mat file using scipy.io and used for rest of the
       operations present in p1_main.py file.
'''
########### Importing Libraries ###########
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as io

from sklearn.datasets import load_breast_cancer

########## Breast Cancer Dataset ##########
data = load_breast_cancer()
X = data['data']
y = data['target']

print 'Size of data matrix =', X.shape
print 'Size of the labels =', y.shape

########### Train and Test Data ###########
# Dividing the data randomly into 70% train 
# and 30% test samples.
sam = X.shape[0]
num = int(0.7*sam)
indcs = range(sam)
random.shuffle(indcs)
train_indcs = indcs[:num]
test_indcs = indcs[num:]
X_train, X_test = X[train_indcs, :], X[test_indcs, :]
y_train, y_test = y[train_indcs], y[test_indcs]

print 'Training data size =', X_train.shape
print 'Training labels size =', y_train.shape
print 'Testing data size =', X_test.shape
print 'Testing labels size =', y_test.shape

io.savemat('custom_data.mat', mdict={'train_data': X_train,
                                     'train_labels': y_train,
                                     'test_data': X_test,
                                     'test_labels': y_test
                                    })