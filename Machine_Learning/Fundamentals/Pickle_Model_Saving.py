# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:43:53 2023

@author: gencs
"""

import pandas as pd 
import numpy as np


url = "https://bilkav.com/satislar.csv"
dataset = pd.read_csv(url)

X = dataset.iloc[:,0:1].values
Y = dataset.iloc[:,1].values

split = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size= split)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train.ravel())

y_pred = lr.predict(X_test)

import pickle 

file = "Saved_Model"

pickle.dump(lr, open(file,"wb"))

loaded_file = pickle.load(open(file,"rb"))

saved_file_pred = loaded_file.predict(X_test)

