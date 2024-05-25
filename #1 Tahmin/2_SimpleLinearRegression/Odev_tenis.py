# -*- coding: utf-8 -*-
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('odev_tenis.csv')

#test
print(veriler)
tempr = veriler.iloc[:,1:3].values

#encoder: Kategorik -> Numeric
outlook = veriler.iloc[:,0:1].values
print(outlook)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veriler.iloc[:,0])

print(outlook)

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

# sunny 2
# rainy 1
# overcast 0


#encoder: Kategorik -> Numeric

windy = veriler.iloc[:,3:4].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,3])

print(windy)

ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()
print(windy)

# false 0
# true 1

#encoder: Kategorik -> Numeric

play = veriler.iloc[:,4:].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

play[:,0] = le.fit_transform(veriler.iloc[:,4])

print(play)

ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()
print(play)

# no 0
# yes 1


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=outlook, index = range(14), columns = ['overcast','rainy','sunny'])
print(sonuc)

#out = veriler.iloc[:,0].values
#print(out)

sonuc2 = pd.DataFrame(data = tempr, index = range(14), columns = ['temperature', 'humidity'])
print(sonuc2)

#wind = veriler.iloc[:,-2].values
#print(wind)

sonuc3 = pd.DataFrame(data=windy[:,-1:], index= range(14), columns= ['windy'])
print(sonuc3)

pla = veriler.iloc[:,-1].values
print(pla)

sonuc4 = pd.DataFrame(data = play[:,-1:], index = range(14), columns= ['play'])
print(sonuc4)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

s3=pd.concat([s2,sonuc4], axis=1)
print(s3)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s3,sonuc4,test_size=0.33, random_state=0)





"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)



hum = s3.iloc[:,4:5].values
print(hum)
sol = s3.iloc[:,:6]

sag = s3.iloc[:,-1:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,hum,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)



import statsmodels.api as sm

#değişkenlerin üzerinde bir model oluşturmak için değişkenleri sisteme ekleme işlemini yapıyoruz
#tahminin sapmasını artıran değerleri backward elimination yapacağız

X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis=1)
#1 lerden oluşan bir dizi başa eklendi (yani sabiti )

X_l = veri.iloc[:,[0,1,2,3,4,5]].values #cıktıda p value yüksek ise bizim için kötü, örnegin 4. veri çıktıda
X_l = np.array(X_l, dtype = float)

model = sm.OLS(boy, X_l).fit() # boy' bağımlı değişken, her kolonun tek tek boy üzerindeki etkisini ölçeceğiz

print(model.summary()) 


"""





