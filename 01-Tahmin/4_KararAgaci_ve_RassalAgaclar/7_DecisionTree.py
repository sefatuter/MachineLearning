#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) # train etmeye yarayan algoritma

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
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

#verilerin olceklenmesi
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

# boy ve kiloyu kullanarak yaşı tahmin edebilir miyiz? 
# decision tree

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X, Y, color = 'red')
plt.plot(X, r_dt.predict(X), color = 'black')

plt.plot(x, r_dt.predict(Z), color = 'green')
plt.plot(x, r_dt.predict(K), color = 'orange')

print(r_dt.predict([[11]])) # 10 dan sonraki herkeste 50000
print(r_dt.predict([[6.6]])) # 7 den önceki hepsi 10000

