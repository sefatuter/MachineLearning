# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:35:18 2024

@author: SefaPc
"""
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. Data pre-processing
#2.1. Data uploading
    
veriler = pd.read_csv("veriler.csv")
#pd.read_csv('veriler.csv') same thing
print(veriler)


x = veriler.iloc[:,1:4].values # bağımsız değişkenler
y = veriler.iloc[:,4:].values  # bağımlı değişkenler






#verilerin eğitim ve test için bölünmesi
#bağımlı ve bağımsız değişken olarak ayırdık
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # x_test için yeniden öğrenme

# fit eğitme, transform ise o eğitimi kullanma


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)

print(y_pred)
print(y_test) # gercek degerler

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)



