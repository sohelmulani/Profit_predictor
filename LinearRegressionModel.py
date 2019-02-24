#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:58:26 2019

@author: sohel
"""

from sklearn.linear_model import LinearRegression

class Regressor(object):
    
    def __init__(self):
        pass
    
    def regressor(self,x,y):
        reg=LinearRegression();
        reg.fit(x,y)
        return(reg)
        
    def __del__(self):
        pass