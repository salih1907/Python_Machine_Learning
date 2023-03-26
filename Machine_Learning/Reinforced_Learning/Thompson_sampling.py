# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 16:31:03 2023

@author: gencs
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")


"""
N = 10000
d = 10
summation = 0
selected = []
for n in range(N):
    ad = random.randrange(d)
    selected.append(ad)
    reward = data.values[n,ad] #if the n th row in data = 1, reward = 1
    summation = summation + reward 
    

plt.hist(selected)
plt.show()
"""
import random
#UCB 

N = 10000 #10000 clicks
d = 10 # total 10 ads
transactions = [0] * d #ad clicks until that time
summation = 0 #total reward 
selected = []
zeros = [0] * d
ones = [0] * d
for n in range(N):
    ad = 0 #selected ad
    max_th = 0
    for i in range(d):
        random_Beta = random.betavariate(ones[i]+1, zeros[i]+1)
        if random_Beta > max_th:
            max_th = random_Beta
            ad = i
        
    selected.append(ad)
    reward = data.values[n,ad] #if the n th row in data = 1, reward = 1
    if reward == 1:
        ones[ad] = ones[ad]+1
    else:
        zeros[ad]= zeros[ad]+1
        
    summation = summation + reward
        
        
print("total reward:")
print(summation)
        
plt.hist(selected)
plt.show()    