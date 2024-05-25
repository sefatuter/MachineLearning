# 1- Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri onisleme
#2.1. Veri yukleme

veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:,0:13].values # Bağımsız değişkenler
Y = veriler.iloc[:,13].values   # Bağımlı değişkenler

# Verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)


# Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train) # İki boyuta indirgendi
X_test2 = pca.transform(X_test)

# PCA dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) # Her logistic regressionda sabit değerle kullan
classifier.fit(X_train, y_train)

# PCA dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# Tahminler

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

# Actual / PCA olmadan çıkan sonuç
cm = confusion_matrix(y_test, y_pred) # y_test orijinal veri ile karşılaştır
print("Gerçek / PCA olmadan çıkan sonuç")
print(cm)

# Actual / PCA sonrası çıkan sonuç
cm2 = confusion_matrix(y_test, y_pred2) # y_test orijinal veri ile karşılaştır
print("Gerçek / PCA sonrası çıkan sonuç")
print(cm2)


# PCA sonrası ve öncesi çıkan sonuç
cm3 = confusion_matrix(y_pred, y_pred2) # y_test orijinal veri ile karşılaştır
print("PCA sonrası ve öncesi çıkan sonuç")
print(cm3)




