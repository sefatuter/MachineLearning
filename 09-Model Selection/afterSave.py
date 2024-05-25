import pandas as pd

url = "http://bilkav.com/satislar.csv" # İnternetten veri çekme
veriler = pd.read_csv(url)
veriler = veriler.values
X = veriler[:, 0:1] # aylardan satışların tahmini
Y = veriler[:,1]

bolme = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=bolme)

import pickle
# Kaydedilen modele erişim
loaded = pickle.load(open("model.save", 'rb'))
print(loaded.predict(X_test))


# Modeli save ettikten sonra sürekli eğitmeye gerek kalmadan çalıştırabilirim