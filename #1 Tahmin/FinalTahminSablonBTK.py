#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score #R^2 - Adjusted R^2 library

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) # train etmeye yarayan algoritma

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

print('Linear R^2 Degeri')
print(r2_score(Y, lin_reg.predict(X)))


#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print('Polynomial R^2 Degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(x))))
# 0.99 gayet iyi deger

#Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))
# y_olcekli = sc1.fit_transform(Y) de olabilir burada

# SVR kodlaması
from sklearn.svm import SVR

svr_reg = SVR(kernel= 'rbf') # radial basis function
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='black')
                    #Her bir x değeri icin tahminde bulun

plt.show() # bu plotu bitir üzerine ekleme yapılmaması için
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

print('SVR R^2 Degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))
#son örnekte ciddi hata degeri 0.75


# boy ve kiloyu kullanarak yaşı tahmin edebilir miyiz? 
# Decision tree

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X, Y, color = 'red')
plt.plot(X, r_dt.predict(X), color = 'black')
plt.plot(x, r_dt.predict(Z), color = 'green')
plt.plot(x, r_dt.predict(K), color = 'orange')
plt.show() # çizimi bitir
print(r_dt.predict([[11]])) # 10 dan sonraki herkeste 50000
print(r_dt.predict([[6.6]])) # 7 den önceki hepsi 10000

print('Decision Tree R^2 Degeri')
print(r2_score(Y, r_dt.predict(X)))


# Random forestta amaç aslında birden fazla karar ağacı oluşturmak
# ve bunların ortalamasından tahmin yapmak

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0) # objeyi oluşturduk, # kaç tane decision tree çizileceği (n estimator)
rf_reg.fit(X, Y.ravel())


print(rf_reg.predict([[6.6]])) # 10500 tahmin etti. Birden fazla tree'den

plt.scatter(X, Y, color = 'red')
plt.plot(X, rf_reg.predict(X), color = 'black')

plt.plot(X, rf_reg.predict(Z), color = 'blue')
plt.plot(X, r_dt.predict(K), color = 'yellow')
plt.show()


print('Random Forest R^2 Degeri')
print(r2_score(Y, rf_reg.predict(X))) # 0.9704434230386582 1'e yakın olması iyi

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))


#Ozet R^2 degerleri
print('R^2 Values'.center(50, "-"))

print('Linear R^2 Degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R^2 Degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(x))))

print('SVR R^2 Degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print('Decision Tree R^2 Degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R^2 Degeri')
print(r2_score(Y, rf_reg.predict(X)))







