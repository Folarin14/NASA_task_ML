# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:51:46 2020

@author: FAHD
"""

import numpy as np, matplotlib.pyplot as plt, os, glob
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# load training and validation dataset; 

# selecting only 3 bands based on Blix et al (442.5, 673.5, 681.25) 
os.chdir('C:\\Users\\FAHD\\Desktop\\NASA task\\Applicant Task')
training = np.genfromtxt('training.csv', delimiter=',', skip_header=1, \
                         usecols=(2,8,9,16))
print(training.shape)
print(training[:5, :])
olci_bands = training[:, :3]
chl = training[:, -1]
#print(olci_bands)
#print(chl)

# subset training data to reduce computation time and memroy usage
x_train, x_test, y_train, y_test = train_test_split(olci_bands, chl, \
                                                    train_size = 0.01, \
                                                        random_state = 20)
print(x_train, x_train.shape)
print(y_train, y_train.shape)

validation = np.genfromtxt('validation.csv', delimiter=',', skip_header=1, \
                           usecols=(2,8,9,16))
print(validation.shape)
print(validation[:5, :])
x_valid = validation[:, :3]
y_valid = validation[:, -1]

# use a SVR algorithm to fit model
model = SVR(C=50000, epsilon=10, cache_size=1000) #gridsearch best param
model.fit(x_train, y_train)

print("\nModel R squared is {} ".format(model.score(x_train, y_train)))

# predict using fitted model 
y_pred = model.predict(x_valid)

# calculate NRMSE
from sklearn.metrics import mean_squared_error, make_scorer
RMSE = mean_squared_error(y_valid, y_pred, squared=False)
N = y_valid.max() - y_valid.min()
NRMSE = RMSE/N

print("\nNormalized Root Mean Squared Error is {}".format(NRMSE))

# load test data and predict Chl-a values
testing = np.genfromtxt('testing.csv', delimiter=',', skip_header=1, \
                           usecols=(2,8,9))


# =============================================================================
# # find optimum/bets paramters using GridSearch
# svr = SVR(cache_size=1000)
# parameters = {'kernel': ['rbf'], 'C': [10000, 20000, 50000, 80000, 100000],\
#               'epsilon': [1, 5, 10, 20, 30]}
#     
# GSC = GridSearchCV(svr, parameters)
# GSC.fit(x_train, y_train)
# 
# print(GSC.get_params())
# #print(sorted(GSC.cv_results_.keys()))
# print(GSC.best_estimator_)
# print(GSC.best_params_)
# print(GSC.best_score_)
# 
# =============================================================================

# using cross validation with 10 splits
#create scoring function NRMSE for CV
def nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    norm = y_true.max() - y_true.min()
    return rmse/norm
score = make_scorer(nrmse, greater_is_better=False)
cross_val_nrmse = cross_val_score(model, x_train, y_train, cv=10, scoring=score)
print(cross_val_nrmse, cross_val_nrmse.mean())
  
#=========================================================================
"""
# selecting only 5 bands based on Blix et al (412.5, 510, 620, 673.5, 681.25) 
os.chdir('C:\\Users\\FAHD\\Desktop\\NASA task\\Applicant Task')
training = np.genfromtxt('training.csv', delimiter=',', skip_header=1, \
                         usecols=(1,4,6,8,9,16))
print(training.shape)
print(training[:5, :])
olci_bands = training[:, :5]
chl = training[:, -1]
#print(olci_bands)
#print(chl)

# subset training data to reduce computation time and memroy usage
x_train, x_test, y_train, y_test = train_test_split(olci_bands, chl, \
                                                    train_size = 0.01, \
                                                        random_state = 200)
print(x_train, x_train.shape)
print(y_train, y_train.shape)

validation = np.genfromtxt('validation.csv', delimiter=',', skip_header=1, \
                           usecols=(1,4,6,8,9,16))
print(validation.shape)
print(validation[:5, :])
x_valid = validation[:, :5]
y_valid = validation[:, -1]

# use a SVR algorithm to fit model
model = SVR(C=50000, epsilon=10, cache_size=1000)
model.fit(x_train, y_train)

print("\nModel R squared is {} ".format(model.score(x_train, y_train)))

# predict using fitted model 
y_pred = model.predict(x_valid)

# calculate NRMSE
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_valid, y_pred, squared=False)
N = y_valid.max() - y_valid.min()
NRMSE = RMSE/N

print("\nNormalized Root Mean Squared Error is {}".format(NRMSE))

# using cross validation with 10 splits
#create scoring function NRMSE for CV
def nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    norm = y_true.max() - y_true.min()
    return rmse/norm
score = make_scorer(nrmse, greater_is_better=False)
cross_val_nrmse = cross_val_score(model, x_train, y_train, cv=10, scoring=score)
print(cross_val_nrmse, cross_val_nrmse.mean())
"""
# 
# SVR(C=50000, epsilon=10, cache_size=1000) #gridsearch best param
# R sq = 0.417, NRMSE = 0.1990, CV 10 fold NRMSE = 0.1444 sample size 7063 3 bands
# R sq = 0.791, NRMSE = 0.2507, CV 10 fold NRMSE = 0.0871 sample size 7063 5 bands
# R sq = 0.931, NRMSE = 0.3165, CV 10 fold NRMSE = 0.0495 sample size 7063 all bands 16

# R sq = 0.449, NRMSE = 0.1702, CV 10 fold NRMSE =  sample size 70633



