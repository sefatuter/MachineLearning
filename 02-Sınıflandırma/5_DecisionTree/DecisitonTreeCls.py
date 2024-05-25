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
print('Logistic Regression'.center(25,'-'))
print(cm)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski') # default hali =5, 'minkowski

knn.fit(X_train, y_train) # önce train et

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('K-NN'.center(25,'-'))
print(cm)

# https://scikit-learn.org/stable/modules/svm.html#support-vector-machines
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('SVC'.center(25,'-'))
print(cm)


# Gaussian Naive Bayes      -> Tahmin etmek istediğimiz veri, sınıf, kolon continuous bir değer ise
# sürekli bir değer ise, reel sayılar olabiliyorsa, ondalık olabiliyorsa

# Multinomial Naive Bayes   -> nominal, birbirinden farklı, araba markası, okuduğun okul nominal değer

# Bernoulli Naive Bayes     -> 1 veya 0, kadın erkek, sigara içiyor içmiyor gibi gibi


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print('GNB'.center(25,'-'))
print(cm)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DecisionTreeCls'.center(25,'-'))
print(cm)






