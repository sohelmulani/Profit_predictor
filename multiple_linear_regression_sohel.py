#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:59:32 2018

@author: sohel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#encoding catagorial data
#encoding trainig data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:, 3] = le.fit_transform(x[:, 3])
ohe = OneHotEncoder(categorical_features = [3])
x = ohe.fit_transform(x).toarray()

#avoiding the dummy variable trap
x=x[:,1:6]

#splitting trainig and testing data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting into multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting values of x_test
y_pred=regressor.predict(x_test)

from sklearn import metrics
accuracy=metrics.explained_variance_score(y_test, y_pred)
print('{}%'.format(accuracy*100))

#predicting optimal solution using backward elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1),dtype=int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

#spliting training and test dataset for x_opt
x_opt_train,x_opt_test,y_train,y_test=train_test_split(x_opt,y,test_size=0.2,random_state=0)

#fitting x_opt_train and y_train in regressor
regressor.fit(x_opt_train,y_train)

#prediction for x_opt_train-
y_opt_pred=regressor.predict(x_opt_test)

from sklearn import metrics
accuracy=metrics.explained_variance_score(y_opt_pred, y_test)
print('{}%'.format(accuracy*100))


