import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("veriler.csv")
missing_data = pd.read_csv("eksikveriler.csv")



heights = dataset[["boy"]]
print(heights)

#missing dataset
#sci - kit learn
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = missing_data.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])


#encoder: Nominal Ordinal (Kategorik) -> Numeric
country = dataset.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(dataset.iloc[:,0])

print(country)


ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)


#numpy dizilieri -> dataframe transformations
result = pd.DataFrame(data=country, index = range(22),columns=["fr","tr","us"])
result2 = pd.DataFrame(data = age, index=range(22),columns=["height","weight","age"])
gender = dataset.iloc[:,-1].values
print(gender)
result3 = pd.DataFrame(data=gender,index= range(22),columns=["gender"])


#dataframe concatenation

s = pd.concat([result,result2],axis=1)
print(s)

s2= pd.concat([s,result3],axis=1)
print(s2)

#TrainTest

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,result3,test_size=0.33,random_state=(0))


#scaling of the dataset
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
 
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)




