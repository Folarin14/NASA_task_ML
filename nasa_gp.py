# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:44:00 2020

@author: FAHD
"""

import numpy as np, matplotlib.pyplot as plt, os, glob
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


# load training and validation dataset; 

# selecting only 3 bands based on Blix et al (442.5, 673.5, 681.25) 
os.chdir('C:\\Users\\FAHD\\Desktop\\NASA task\\Applicant Task')
training = np.genfromtxt('training.csv', delimiter=',', skip_header=1, \
                         usecols=(2,8,9,16))
print(training.shape)
print(training[:5, :])
olci_bands = training[:, :3]
chl = training[:, -1]
print(olci_bands)
print(chl)

# subset training data to reduce computation time and memroy usage
x_train, x_test, y_train, y_test = train_test_split(olci_bands, chl, \
                                                    train_size = 0.01, \
                                                        random_state = 44)
print(x_train, x_train.shape)
print(y_train, y_train.shape)

validation = np.genfromtxt('validation.csv', delimiter=',', skip_header=1, \
                           usecols=(2,8,9,16))
print(validation.shape)
print(validation[:5, :])
x_valid = validation[:, :3]
y_valid = validation[:, -1]
#print(x_valid)
#print(y_valid)
"""
kernel1 = gp.kernels.ConstantKernel(1.0) * gp.kernels.RBF(length_scale=1.0)
kernel2 = gp.kernels.ConstantKernel(10.0) * gp.kernels.RBF(length_scale=10.0)
#'kernel': [kernel1, kernel2],

model = gp.GaussianProcessRegressor(kernel1)
parameters = {'alpha': [1e-10, 1e-5, 0.1, 1]}
    
GSC = GridSearchCV(model, parameters)
GSC.fit(x_train, y_train)

print(GSC.get_params())
#print(sorted(GSC.cv_results_.keys()))
print(GSC.best_estimator_)
print(GSC.best_params_)
print(GSC.best_score_)
"""

# use a guassian process regression algo to fit model
# define kernel

kernel = gp.kernels.ConstantKernel() * gp.kernels.RBF(length_scale=10.0)
model_gpr = gp.GaussianProcessRegressor(kernel= kernel,alpha=1000, \
                                    n_restarts_optimizer=0, \
                                    normalize_y=False, random_state=44)
model_gpr.fit(x_train, y_train)

print("Model parameters are \n{} ".format(model_gpr.get_params()))
print("\nModel Kernel parameters are \n{} ".format(model_gpr.kernel_.get_params()))
print("\nModel R squared is {} ".format(model_gpr.score(x_train, y_train)))

# predict using fitted model 
y_pred, std = model_gpr.predict(x_valid, return_std = True)

# calculate NRMSE
from sklearn.metrics import mean_squared_error, make_scorer
RMSE = mean_squared_error(y_valid, np.absolute(y_pred), squared=False)
N = y_valid.max() - y_valid.min()
NRMSE = RMSE/N

print("\nNormalized Root Mean Squared Error is {}".format(NRMSE))

"""
# using cross validation with 10 splits
# create scoring function NRMSE for CV
def nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    norm = y_true.max() - y_true.min()
    return rmse/norm
score = make_scorer(nrmse, greater_is_better=False)
cross_val_nrmse = cross_val_score(model, x_train, y_train, cv=10, scoring=score)
print(cross_val_nrmse, cross_val_nrmse.mean())
"""
#=========================================================================
"""
# selecting only 5 bands based on Blix et al (442.5, 673.5, 681.25) 
os.chdir('C:\\Users\\FAHD\\Desktop\\NASA task\\Applicant Task')
training = np.genfromtxt('training.csv', delimiter=',', skip_header=1, \
                         usecols=(1,4,6,8,9,16))
print(training.shape)
print(training[:5, :])
olci_bands = training[:, :5]
chl = training[:, -1]
print(olci_bands)
print(chl)

# subset training data to reduce computation time and memroy usage
x_train, x_test, y_train, y_test = train_test_split(olci_bands, chl, \
                                                    train_size = 0.01, \
                                                        random_state = 44)
print(x_train, x_train.shape)
print(y_train, y_train.shape)

validation = np.genfromtxt('validation.csv', delimiter=',', skip_header=1, \
                           usecols=(1,4,6,8,9,16))
print(validation.shape)
print(validation[:5, :])
x_valid = validation[:, :5]
y_valid = validation[:, -1]
#print(x_valid)
#print(y_valid)

# use a guassian process regression algo to fit model
# define kernel

kernel = gp.kernels.ConstantKernel() * gp.kernels.RBF(length_scale=10.0)
model = gp.GaussianProcessRegressor(kernel= kernel,alpha=1e-10, \
                                    n_restarts_optimizer=0, \
                                    normalize_y=False, random_state=44)
model.fit(x_train, y_train)

print("Model parameters are \n{} ".format(model.get_params()))
print("\nModel Kernel parameters are \n{} ".format(model.kernel_.get_params()))
print("\nModel R squared is {} ".format(model.score(x_train, y_train)))

# predict using fitted model 
y_pred, std = model.predict(x_valid, return_std = True)

# calculate NRMSE
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_valid, y_pred, squared=False)
N = y_valid.max() - y_valid.min()
NRMSE = RMSE/N

print("\nNormalized Root Mean Squared Error is {}".format(NRMSE))
"""








