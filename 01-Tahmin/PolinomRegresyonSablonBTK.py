#Polynomial Regression Sablon

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

# Data Frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# Numpy array dönüşümü
X = x.values
Y = y.values


#Linear Regression
#Doğrusal model oluşturma (Linear Model)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Polynomial Regression
#Doğrusal olmayan model oluşturma (Non - Linear model)
#2. Dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4. Dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


#Grafik görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[11]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print(lin_reg3.predict(poly_reg3.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg3.fit_transform([[11]])))











