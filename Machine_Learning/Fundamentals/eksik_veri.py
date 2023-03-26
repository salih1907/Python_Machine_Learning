# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 18:35:10 2022

@author: gencs
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#
dataset = pd.read_csv("veriler.csv")
missing_data = pd.read_csv("eksikveriler.csv")


print(dataset)
print(missing_data)

heights = dataset[["height"]]
print(heights)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = missing_data.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)