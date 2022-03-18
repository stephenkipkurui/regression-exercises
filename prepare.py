from wrangle import wrangle_zillow
from operator import imod
import env
import pandas as pd
import numpy as np
import os
import sklearn.preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler



def min_max_scaller():
   
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    
    train, validate, test = wrangle_zillow()
    min_max_scaler.fit(train)
    x_train_min_max_scaller = min_max_scaler.transform(train)
    x_validate_min_max_scaller = min_max_scaler.transform(validate)
    x_test_min_max_scaller = min_max_scaler.transform(test)
    
    return x_train_min_max_scaller, x_validate_min_max_scaller, x_test_min_max_scaller
    

def standard_scaller():

    std_scaler = sklearn.preprocessing.StandardScaler()
    train, validate, test = wrangle_zillow()
    min_max_scaler.fit(train)
    x_train_std_scaller = std_scaler.transform(train)
    x_validate_std_scaller = std_scaler.transform(validate)
    x_test_std_scaller = std_scaler.transform(test)
    
    return x_train_std_scaller, x_validate_std_scaller, x_test_std_scaller

    
def robust_scaller():
    
    robust_scaler = sklearn.preprocessing.RobustScaler()
    train, validate, test = wrangle_zillow()
    robust_scaler.fit(train)
    x_train_std_scaller = robust_scaler.transform(train)
    x_validate_std_scaller = robust_scaler.transform(validate)
    x_test_std_scaller = robust_scaler.transform(test)
    
    return x_train_std_scaller, x_validate_std_scaller, x_test_std_scaller

    
    