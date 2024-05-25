
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#encoder: Kategorik -> Numeric

#COUNTRY İÇİN -------------data

ulke = veriler.iloc[:, 0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() #sayısal olarak her bir değere atama yapılmasını sağlar. 

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

#ülke kolonunu 0 1 ve 2 olarak bölümledik + test ettik print ile
print(ulke)

#ülke kolonunu sayısal değerlere dönüştürdük
ohe = preprocessing.OneHotEncoder() #kolon başlıklarını etiketlere taşımak  1 veya 0 yaparak oraya ait mi değil mi yapar.
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
#ülke kolonunu TR, FR, US var ya da yok 1 0 şeklinde dönüştürdük
#Kategorik verinin SAYISAL veriye nasıl çevrildiğini gördük


# CİNSİYET İÇİN-----------
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)



#Verilerin birleştirilmesi, numpy dizileri dataframe dönüşümü
result = pd.DataFrame(data=ulke, index = range(22), columns = ['fr', 'tr', 'us'])
print(result)

result2 = pd.DataFrame(data=veriler, index = range(22), columns = ['boy', 'kilo', 'yas'])
print(result2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

result3 =pd.DataFrame(data=c[:,:1], index = range(22), columns=['cinsiyet'])
print(result3)


#dataframe birleştirme işlemi
s = pd.concat([result,result2], axis = 1)
print(s)

s2 = pd.concat([s, result3], axis = 1)
print(s2)

#verilerin eğitim ve test için bölünmesi
#bağımlı ve bağımsız değişken olarak ayırdık
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, result3, test_size=0.33, random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#amacımız ülke kilo ve yaştan cinsiyet belirleme

















