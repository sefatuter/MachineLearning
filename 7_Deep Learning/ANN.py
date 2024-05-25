# Müşteri kayıp analizi.

# 1- Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri onisleme
#2.1. Veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')

print(veriler)

X = veriler.iloc[:,3:13].values # Bağımsız değişkenler
Y = veriler.iloc[:,13].values   # Bağımlı değişkenler


# Encoder: Kategorik -> Numeric (Gender and Country)
from sklearn import preprocessing

le = preprocessing.LabelEncoder() # Coğrafi kısım
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder() # Gender kısmı
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]


# Verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

# Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#3- Yapay Sinir ağı
import keras
from keras.models import Sequential   #
from keras.layers import Dense        # Katman oluşturma

classifier = Sequential() 

# Giriş nöronu 11 tane çünkü 11 tane bağımsız değişkenimiz var
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu' , input_dim = 11))
# 6 Nörondan oluşan gizli katman (Hidden Layer)
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
# 6 Nörondan oluşan ikinci gizli katman (Hidden Layer)
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# 1 Nörondan oluşan çıkış katmanı
# Katmandaki nöron sayısı giriş (nöron + çıkış nöron / 2)

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train, y_train, epochs=50) # bağımsız değişkenler X den y'yi öğren

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5) # Doğrular 1 yanlıslar 0 yaptık bize tam değer lazım 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)  # gerçek veriler, y_pred tahminler

print(cm)













