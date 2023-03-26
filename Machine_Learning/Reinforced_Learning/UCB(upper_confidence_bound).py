# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:04:25 2023

@author: gencs
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")

import random
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
import math
#UCB 

N = 10000 #10000 clicks
d = 10 # total 10 ads
#Ri(n)
rewards = [0] * d #the reward of the ads is 0 at the beginning
#Ni(n)
transactions = [0] * d #ad clicks until that time
summation = 0 #total reward 
selected = []
for n in range(N):
    ad = 0 #selected ad
    max_ucb = 0
    for i in range(d):
        if transactions[i] > 0:    
            average = rewards[i] / transactions[i]
            delta = math.sqrt(3/2* math.log(n)/transactions[i])
            ucb = average + delta 
        else:
            ucb = N*10
            
        if max_ucb < ucb: #if have a higher ucb value
            max_ucb = ucb
            ad = i
        
    selected.append(ad)
    transactions[ad] = transactions[ad]+1
    reward = data.values[n,ad] #if the n th row in data = 1, reward = 1
    rewards[ad] = rewards[ad] + reward
    summation = summation + reward
        
        
print("total reward:")
print(summation)
        
plt.hist(selected)
plt.show()    
        