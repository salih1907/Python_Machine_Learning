# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:12:06 2023

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
plt.show()


kmeans = KMeans(n_clusters= 4, init="k-means++",random_state=(123)) 
kmeans.fit(x)
Y_pred = kmeans.predict(x)
plt.scatter(x[Y_pred==0,1],x[Y_pred==0,2], s=100,color="red")
plt.scatter(x[Y_pred==1,1],x[Y_pred==1,2], s=100,color="blue")
plt.scatter(x[Y_pred==2,1],x[Y_pred==2,2], s=100,color="green")
plt.scatter(x[Y_pred==3,1],x[Y_pred==3,2], s=100,color="purple")
plt.title("KMeans")
plt.show()

#hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean",linkage = "ward")
Y_pred = ac.fit_predict(x)

plt.scatter(x[Y_pred==0,1],x[Y_pred==0,2], s=50,color="red")
plt.scatter(x[Y_pred==1,1],x[Y_pred==1,2], s=50,color="blue")
plt.scatter(x[Y_pred==2,1],x[Y_pred==2,2], s=50,color="green")
plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch 
dendogram = sch.dendrogram(sch.linkage(x,method= "ward"))
plt.show()








