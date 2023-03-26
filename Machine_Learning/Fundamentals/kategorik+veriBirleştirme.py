# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 23:32:24 2022

@author: gencs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("veriler.csv")
missing_data = pd.read_csv("eksikveriler.csv")


print(dataset)
print(missing_data)

heights = dataset[["boy"]]
print(heights)

from sklearn.impute import SimpleImputer
from sklearn import preprocessing

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = missing_data.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])

country = dataset.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(dataset.iloc[:,0])

print(country)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)

result = pd.DataFrame(data=country, index = range(22),columns=["fr","tr","us"])
result2 = pd.DataFrame(data = age, index=range(22),columns=["boy","kilo","age"])
gender = dataset.iloc[:,-1].values
print(gender)
result3 = pd.DataFrame(data=gender,index= range(22),columns=["cinsiyet"])
s = pd.concat([result,result2],axis=1)
print(s)

s2= pd.concat([s,result3],axis=1)
print(s2)




