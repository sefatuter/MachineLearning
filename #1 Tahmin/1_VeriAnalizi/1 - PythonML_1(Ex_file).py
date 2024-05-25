"""
Created on Sat Mar 30 13:35:18 2024

@author: SefaPc
"""
#lesson 6: Libraries

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. Data pre-processing
#2.1. Data uploading
data = pd.read_csv("eksikveriler.csv")
#pd.read_csv('veriler.csv') same thing

print(data)

#data processing

heightWeight = data[['boy', 'kilo']]

print(heightWeight)
print("\n")
x = 10

class human:
    height = 180
    def run(self,b):
        return b + 10
    # y = f(x)
    # f(x) = x + 10
    
ali = human()
print(ali.height)
print(ali.run(100))

list = [1,2,3]
print(list)


#sci - kit learn
#Missing Data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

yas = data.iloc[:,1:4].values

print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)

country = data.iloc[:, 0:1].values
print(country)

## Ders 2_6 Baslangıc

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(data.iloc[:,0])

#ülke kolonunu 0 1 ve 2 olarak bölümledik
print(country)

#ülke kolonunu sayısal değerlere dönüştürdük

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)
#ülke kolonunu TR, FR, US var ya da yok 1 0 şeklinde dönüştürdük
#Kategorik verinin SAYISAL veriye nasıl çevrildiğini gördük

#Verilerin birleştirilmesi;
result = pd.DataFrame(data=country, index = range(22), columns = ['fr', 'tr', 'us'])
print(result)

result2 = pd.DataFrame(data=yas, index = range(22), columns = ['boy', 'kilo', 'yas'])

gender = data.iloc[:,-1].values
print(gender)

result3 =pd.DataFrame(data=gender, index = range(22), columns=['cinsiyet'])
print(result3)

s = pd.concat([result,result2], axis = 1)
print(s)

s2 = pd.concat([s, result3], axis = 1)
print(s2)

#ülke, boy, kilo ve yaştan cinsiyetin tahmin edilebilmesini istiyoruz

#bağımlı ve bağımsız değişken olarak ayırdık
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, result3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)




