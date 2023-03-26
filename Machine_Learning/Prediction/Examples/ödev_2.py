# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:02:15 2023

@author: gencs
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
#2. Veri Onisleme
#2.1. Veri Yukleme
veriler = pd.read_csv('maaslar_yeni.csv')
check = pd.read_csv("maaslar_yeni.csv")

x= veriler.iloc[:,2:3]
y= veriler.iloc[:,5:]
X= x.values
Y= y.values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

import statsmodels.api as sm

print("Linear r2 score")
model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())

print(r2_score(Y, lin_reg.predict(X)))


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

print("Polynomial r2 score")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)), X)
print(model2.fit().summary())


print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_scaled = sc.fit_transform(X)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y)



#SVR regression
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaled, y_scaled.ravel())

print("SVR r2 score")
model3 = sm.OLS(svr_reg.predict(x_scaled), x_scaled)
print(model3.fit().summary())


print(r2_score(y_scaled, svr_reg.predict(x_scaled)))


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=(0))
r_dt.fit(X,Y.ravel())

print("decision tree r2 score")
model4 = sm.OLS(r_dt.predict(X), X)
print(model4.fit().summary())


print(r2_score(Y, r_dt.predict(X)))


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators= 10, random_state = 0)
rf_reg.fit(X, Y.ravel())


print("Random forest r2 score")
model5 = sm.OLS(rf_reg.predict(X), X)
print(model5.fit().summary())


print(r2_score(Y, rf_reg.predict(X)))



#tahminler 
print(lin_reg.predict([[10]]))
print(lin_reg2.predict(poly_reg.fit_transform([[10]])))
print(svr_reg.predict(sc.fit_transform([[10]])))
print(r_dt.predict([[10]]))
print(rf_reg.predict([[10]]))

#Extra yöntem
print(veriler.corr())


