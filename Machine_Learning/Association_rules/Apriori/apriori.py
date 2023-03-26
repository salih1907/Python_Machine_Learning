# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:50:09 2023

@author: gencs
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

Data = pd.read_csv("sepet.csv" ,header = None)

t = []

for i in range(0,7501):
    t.append([str(Data.values[i,j]) for j in range(0,20)])

from apyori import apriori
rules = list(apriori(t, min_support = 0.01, min_confidence= 0.2,min_lift =3, min_length=2))
print(rules)