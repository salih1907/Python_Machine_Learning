# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:50:16 2023

@author: gencs
"""


import pandas as pd
import numpy as np



dataset = pd.read_csv("veriler.csv")
missing_data = pd.read_csv("eksikveriler.csv")



heights = dataset[["boy"]]
print(heights)

#missing data
#sci - kit learn
from sklearn.impute import SimpleImputer


imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = missing_data.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])


#encoder: Nominal Ordinal (Kategorik) -> Numeric
from sklearn import preprocessing
country = dataset.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(dataset.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)



gender = dataset.iloc[:,-1:].values
le = preprocessing.LabelEncoder()

gender[:,-1] = le.fit_transform(dataset.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()
print(gender)
 

#numpy dizilieri -> dataframe transformations
result = pd.DataFrame(data=country, index = range(22),columns=["fr","tr","us"])
result2 = pd.DataFrame(data = age, index=range(22),columns=["height","weight","age"])
genders = dataset.iloc[:,-1].values
result3 = pd.DataFrame(data=gender[:,:1],index= range(22),columns=["gender"])


#dataframe concat

s = pd.concat([result,result2],axis=1)
print(s)

s2= pd.concat([s,result3],axis=1)
print(s2)

#train-test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,result3,test_size=0.33,random_state=(0))


#data scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
 
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#REGRESSION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

height = s2.iloc[:,3:4].values
print(height)

left = s2.iloc[:, :3]
right = s2.iloc[:, 4:]

data = pd.concat([left,right],axis = 1)

x_train ,x_test, y_train, y_test = train_test_split(data,height,test_size=0.33,random_state=(0))

r2 = LinearRegression()
r2.fit(x_train, y_train)


y_pred = r2.predict(x_test)
print(type(x_test))

#BACKWARD ELIMINATION
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values= data, axis = 1)

X_list = data.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list,dtype=(float))
model = sm.OLS(height, X_list).fit()
print(model.summary())

X_list = data.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list,dtype=(float))
model = sm.OLS(height, X_list).fit()
print(model.summary())





































