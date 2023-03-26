# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:15:40 2023

@author: gencs
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv("musteriler.csv")

from sklearn.preprocessing import LabelEncoder  

le = LabelEncoder()
dataset2 = le.fit_transform(dataset.iloc[:,1:2])
dataset2 = pd.DataFrame(dataset2)

x = dataset.iloc[:,2:4]
x= pd.concat([dataset2,x],axis=1).values
y = dataset.iloc[:,4:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 3, init="k-means++") 
kmeans.fit(x)
print(kmeans.cluster_centers_)
results = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init="k-means++",random_state=(123)) 
    kmeans.fit(x)
    results.append(kmeans.inertia_)
    
plt.plot(range(1,11), results)