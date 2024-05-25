"""
odev2
@author: Sefa
"""

# Gerekli / Gereksiz bağımsız değişkenleri bulunuz
    # P- Value Kullanabilirsin

# 5 Farklı Yönteme göre regresyon modellerini çıkarınız
    # MLR, PR, SVR, DT, RF
        # MULTIPLE LINEER REGRESSION
        # POLYNOMIAL REGRESSION
        # SUPPORT VECTOR REGRESSION
        # DECISION TREE
        # RANDOM FOREST
# Yöntemlerin başarılarını karşılaştır. (R^2 Yöntemi veya Adjusted R^2 Yöntemi )

# 10 yıl tecrübeli ve 100 puan almış bir CEO ve aynı özelliklere

# sahip bir Müdürün maaşlarını 5 yöntemle de tahmin edip sonuçları yorumlayınız


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score #R^2 - Adjusted R^2 library

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

x = veriler.iloc[:,2:5] # UnvanSeviyesi, Kidem, Puan
y = veriler.iloc[:,5:] # Maas
 # ID kolonu alınmaz, Unvan alınmaz (dummy variable)

X = x.values
Y = y.values

"""
calisanID = veriler.iloc[:,:1]
veriler2 = pd.concat([x,y], axis=1)
veriler3 = pd.concat([veriler2, calisanID], axis=1)
print(veriler3.corr())

#correlation matrisi bu tabloda cıkan değerler birbirleri arasındaki ilişkiyi gösterir
# bu tabloya bakarak hangilerinin birbiriyle tahmini için daha net kullanıldığını görebiliriz.
# örneğin kidem ile kidem arasındaki ilişki 1.00000
# calisan id ile puan arasındaki ilişki -0.251278 yani çok kötü alakasız demek
# maasla en yüksek ilişkiyi veren UnvanSeviyesi görüldüğü gibi

ödevin kısa yolu olarak düsünülebilir.
el ile çözdüğümüzde de UnvanSeviyesi ve Kidem'in çıkarılmasının tahmine pozitif yönde etki yaptığını gözlemledik
"""


# P- Value

#Bağımsız değişkenlerden bağımlı değişken bulunur.
# Gereksiz değişkenler -> UnvanSeviyesi, Kidem, Puan (Bağımsız değişkenler) X 
# Gerekli değişkenler -> Maas (Bağımlı değişkenler) Y

#Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) # train etmeye yarayan algoritma

import statsmodels.api as sm

print('\n')
print('Linear OLS'.center(50, '-'))
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

# x = veriler.iloc[:,2:5] ' den [:,2:3] yaptık yani UnvanSeviyesi ve Kidem bağımsız değişkenleri sistemi olumsuz etkiliyormuş
#   R-square değeri 1. durumda 0.903 iken 2. durumda 0.942 oldu
#tahminin sapmasını artıran değerleri backward elimination yapacağız


#Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


print('\n')
print('Polynomial OLS'.center(50, '-'))
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())


# SVR kodlaması

#Verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))
# y_olcekli = sc1.fit_transform(Y) de olabilir burada


from sklearn.svm import SVR

svr_reg = SVR(kernel= 'rbf') # radial basis function
svr_reg.fit(x_olcekli, y_olcekli)


print('\n')
print('SVR OLS'.center(50, '-'))
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())


print('SVR R^2 Degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


#Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4


print('\n')
print('Decision Tree OLS'.center(50, '-'))
model4  = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())


print('Decision Tree R^2 Degeri')
print(r2_score(Y, r_dt.predict(X)))


#Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0) # objeyi oluşturduk, # kaç tane decision tree çizileceği (n estimator)
rf_reg.fit(X, Y.ravel())


print('\n')
print('Random Forest OLS'.center(50, '-'))
model5  = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary()),


print('Random Forest R^2 Degeri')
print(r2_score(Y, rf_reg.predict(X))) # 0.9704434230386582 1'e yakın olması iyi

print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))


"""
x = veriler.iloc[:,2:3] iken "UnvanSeviyesi", tek parametreli olarak;

LINEAR  R-squared (uncentered):                   0.942
POLY    R-squared (uncentered):                   0.759
SVR     R-squared (uncentered):                   0.770
DT      R-squared (uncentered):                   0.751
RF      R-squared (uncentered):                   0.719


x = veriler.iloc[:,2:5] iken yani  "UnvanSeviyesi, Kidem, Puan" Kidem ve Puan parametreleri eklendiğinde, 3 parametreli olarak;

LINEAR  R-squared (uncentered):                   0.903
POLY    R-squared (uncentered):                   0.680
SVR     R-squared (uncentered):                   0.782
DT      R-squared (uncentered):                   0.679
RF      R-squared (uncentered):                   0.713


2:5 den 2:3 yaptığımızda yani kidem ve puan parametreleri elendiğinde R^2 tarafından bakıldığında olumlu 

linear artmış (iyileşme)
poly artmış (iyileşme)
svr azalmış
dt artmış (iyileşme)
rd  artmış (iyileşme)


"""
























