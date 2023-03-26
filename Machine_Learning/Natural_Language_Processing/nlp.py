# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:06:38 2023

@author: gencs
"""

import pandas as pd 
import numpy as np

comments = pd.read_csv("Restaurant_Reviews.csv" ,
                       on_bad_lines=("skip"),
                       keep_default_na=False)
                      

#PREPROCESSING

import re #Regular Expressions
import nltk

from nltk.stem.porter import PorterStemmer #for root of the word
ps = PorterStemmer()

#nltk.download("stopwords")
from nltk.corpus import stopwords


count_row = comments.shape[0]

corpus = []
for i in range(count_row):
    comment = re.sub("[^a-zA-Z]"," ",comments["Review"][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words("english"))]
    comment = " ".join(comment)
    corpus.append(comment)

#FEATURE EXTRACTION
#Bag of words (BOW)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=(2000))
X = cv.fit_transform(corpus).toarray()  #independent variable
Y = comments.iloc[:,1].values           #dependent variable


#Machine learning

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=(0))

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train.ravel())

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)

print(cm)



