#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#2. Data pre-processing
#2.1. Data uploading
    
veriler = pd.read_csv("iris.csv")
print(veriler)


x = veriler.iloc[:,0:4].values # bağımsız değişkenler
y = veriler.iloc[:,4:5].values  # bağımlı değişkenler


#verilerin eğitim ve test için bölünmesi
#bağımlı ve bağımsız değişken olarak ayırdık
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
# X_train: Modelin eğitildiği veri setindeki bağımsız değişkenler.
# X_test: Modelin performansının değerlendirildiği, 
#  eğitim sırasında kullanılmayan veri setindeki bağımsız değişkenler.



# Buradan itibaren sınıflandırma algoritmaları başlar.
#---> 1. Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train) # Eğitim .fit()

y_pred = logr.predict(X_test) # Tahmin .predict()
# print(y_pred) # tahmin edilen degerler
# print(y_test) # gercek degerler


# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)
print('Logistic Regression'.center(25,'-'))
print(cm)


#---> 2. KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski') # default hali =5, 'minkowski'
knn.fit(X_train, y_train) # önce train et

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('K-NN'.center(25,'-'))
print(cm)

#---> 3. SVC
# https://scikit-learn.org/stable/modules/svm.html#support-vector-machines
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('SVC'.center(25,'-'))
print(cm)



#---> 4. Naive Bayes

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


#---> 5. Decisiton Tree

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="gini")
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DecisionTreeCls'.center(25,'-'))
print(cm)


#---> 6. Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
y_proba = rfc.predict_proba(X_test) # tahmin olasılıkları
cm = confusion_matrix(y_test, y_pred)
print('RandomForestCls'.center(25,'-'))
print(cm)


#---> 7. ROC
#print(y_test)
#print(y_proba[:,0]) # Confidence intervals

from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')

print('False Positive Rates = ', fpr)
print('True Positive Rate = ', tpr)
print('Treshold = ', thold)
