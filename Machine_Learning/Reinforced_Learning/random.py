# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:15:36 2023

@author: gencs
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")

import random

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