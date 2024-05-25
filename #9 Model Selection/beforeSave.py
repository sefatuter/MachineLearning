import pandas as pd

url = "http://bilkav.com/satislar.csv" # İnternetten veri çekme
veriler = pd.read_csv(url)
veriler = veriler.values
X = veriler[:, 0:1] # aylardan satışların tahmini
Y = veriler[:,1]

bolme = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=bolme)

# Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)

print(lr.predict(X_test))

import pickle

file = "Model.save"
pickle.dump(lr, open(file,"wb")) # Kaydedilen veriler diske kaydedilecek

# Kaydedilen modele erişim
loaded = pickle.load(open(file, 'rb'))
print(loaded.predict(X_test))

