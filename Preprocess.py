#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:40:04 2019

@author: sohel
"""
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

class Encoder(object):
    def __init__(self):
        pass
    
    def encoder(self,data):
        Edata=pd.get_dummies(data)
        return(Edata)
        
    def __del__(self):
        pass
    
class Splitter(object):
    def __init__(self):
        pass
    
    def decomposition(self,x,y,test_size=0.2,random_state=0):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
        return(x_train,x_test,y_train,y_test)
    
    def __del__(self):
        pass
        
        
        
        
        
        