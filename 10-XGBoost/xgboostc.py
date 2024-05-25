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


from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
print(cm)

