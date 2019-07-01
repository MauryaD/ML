# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:24:16 2019

@author: Deepak Maurya
"""

#DATASET PREPROCESSING
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset= pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values    # matrix of future...independent variables
Y=dataset.iloc[:,3].values      # dependent variable

#for missing data
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.NaN, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding catergorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

#trainung set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#scaling
from sklearn.preprocessing import StandardScaler
sl_X = StandardScaler()
x_train = sl_X.fit_transform(x_train)
x_test = sl_X.transform(x_test)