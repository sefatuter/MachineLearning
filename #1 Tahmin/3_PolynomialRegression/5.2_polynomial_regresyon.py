#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri yükleme
veriler = pd.read_csv('maaslar.csv')

# Verilerin içinden dataframe olarak kolonları aldık.
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,-1:]

# X = x.values
# Y = y.values hata alırsan kullan



#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x, y) # fit ->  x den y yi öğren

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'black')
plt.show()

#Polynomial Regression


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # polinom objesi oluştur 2. dereceden
x_poly = poly_reg.fit_transform(x) # x değerini x poly olarak alıcaz

print(x_poly)

# x^1 + x^n + x^(n^2) şeklinde ikinci dereceden fazla üssü alınarak giden denklem

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y) # değişkenleri kullanarak y'yi öğren

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = 'black') # burada polinomala dönüştürdük
plt.show()


#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]])) #linear için

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]]))) #polynomial için daha isabetli





