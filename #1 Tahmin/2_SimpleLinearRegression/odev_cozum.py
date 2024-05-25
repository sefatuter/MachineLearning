
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')

print(veriler)

"""
1.yol

#encoder: Kategorik -> Numeric

play = veriler.iloc[:, -1:].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() #sayısal olarak her bir değere atama yapılmasını sağlar. 0 lar ve 1 ler olarak

play[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(play)


#encoder: Kategorik -> Numeric

windy = veriler.iloc[:, -2:-1].values
print(windy)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() #sayısal olarak her bir değere atama yapılmasını sağlar. 0 lar ve 1 ler olarak

windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(windy)

"""

#kısa yol 2. yol
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)


c = veriler2.iloc[:,:1].values

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

#hava durumu için one hot encoding 0 1 ama 2 den fazla durum var
#1 ve 0 a dönüşümler için label encoding

# sunny 2
# rainy 1
# overcast 0

#Verilerin birleştirilmesi, numpy dizileri dataframe dönüşümü

havadurumu = pd.DataFrame(data = c, index = range(14), columns = ['ovr', 'rai', 'sun'])

sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)

#sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis = 1) # birleştirilmiş hali
#ilk 3 one hot encodingten gelen geri kalanı orijinal veri dosyasından
#sonveriler = pd.concat([sonveriler, veriler2.iloc[:,-2:]], axis = 1)
#son kısmı da yine orijinal dosyadan alıp birleştirdik

#deneme = pd.concat([sonveriler.iloc[:,:4], sonveriler.iloc[:,5:7]], axis=1)

#diyelim ki humidity tahmin etmeye çalışacağız: 
# bu durumda humidity bağımlı değişken geri kalanlar bağımsız değişken

#verilerin eğitim ve test için bölünmesi
#bağımlı ve bağımsız değişken olarak ayırdık

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1], sonveriler.iloc[:,-1:], test_size=0.33, random_state=0)

# sonveriler.iloc[:,:-1] başlangıçtan son kolon hariç al, sonveriler.iloc[:,-1:] son kolon da bağımlı değişken

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



# humidity tahmini çözüm buraya kadardı. şimdi backward elimination metodu ile geliştir
#BACKWARD ELIMINATION odev devami

import statsmodels.api as sm

#değişkenlerin üzerinde bir model oluşturmak için değişkenleri sisteme ekleme işlemini yapıyoruz
#tahminin sapmasını artıran değerleri backward elimination yapacağız

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values #cıktıda p value yüksek ise bizim için kötü
X_l = np.array(X_l, dtype = float)

model = sm.OLS(sonveriler.iloc[:,-1:], X_l).fit()

print(model.summary()) 

# x1 en büyük olduğu için x1 i backward elimination
sonveriler = sonveriler.iloc[:,1:] #windy kolonu silindi

import statsmodels.api as sm


X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values #cıktıda p value yüksek ise bizim için kötü
X_l = np.array(X_l, dtype = float)

model = sm.OLS(sonveriler.iloc[:,-1:], X_l).fit() 

print(model.summary()) 

#x train ve x testten de windyi atıyoruz

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

# ve yeni train değeriyle sistemi tekrar eğit

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#eski y_pred değerlere göre daha gerçeğe yakın tahmin yapıldığı görülebilir.


"""

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())



"""

 