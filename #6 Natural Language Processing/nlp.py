import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split


# yorum = re.sub('[^a-zA-Z]',' ', yorumlar['Review'][0])# ^ not anlamında
# # İlk satırdaki yorumu aldık noktalama işaretlerini temizleyeceğiz

# Stop words problemi - temizleme işlemi
nltk.download('stopwords')

# kelimelerin köklerini bulma problemi
ps = PorterStemmer()
nltk.download('stopwords')


# Preprocessing (Önişleme)
derlem = []

for i in range(406):
    yorum = re.sub('[^a-zA-Z]',' ', yorumlar['Review'][i])# ^ not anlamında
    # İlk satırdaki yorumu aldık noktalama işaretlerini temizleyeceğiz
    
    # Büyük-Küçük harf problemi
    yorum = yorum.lower()
    yorum = yorum.split()
    
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] # kelimelerin köklerini alacak.
    # stopwords import'undan hazır stopword'leri alır 'turkish' için türkçe stopwordleri alır.
    # şayet kelime stopword değilse bunu stemle yani kökünü al ve listeye at
    
    # Tekrar cümleye çeviriyoruz
    yorum = ' '.join(yorum) # yorumu al boşluklarla birleştir stringe dönüştür
    # Artık bu stringi işleyebiliriz for ile
    derlem.append(yorum) # çıkan stringleri listeledik
    
# sci kit learn ile machine learning ile --> Feature Extraction ( Öznitelik Çıkarımı ) işlemine geçiyoruz
# Bag of Words (BOW)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000) # en fazla kullanılan 2000 kelimeyi al dedik

X = cv.fit_transform(derlem).toarray() # Bağımsız değişken, hem train et hem de uygula
# 2000 dediğimiz kelimeler için o kelime bizim derlemimizde var mı yok mu? bakıyoruz
y = yorumlar.iloc[:,1].values # Bağımlı değişken


# Machine Learning kısmı.

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)
# %80'e %20 böl, %80 eğit, %20 ile başarıyı test et

# X_train'den y_train'i öğren
# X_test'tekileri tahmin et, y_testtekilerle karşılaştır

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test) # yorumlardaki 1 ve 0'ları tahmin ettiriyoruz.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # tahminlerle gerçek değerleri confusion matrixe aktardık

print(cm) # 50/80*100 %60 civarı doğruluk oranı 400 verinin % 80 sinin test edilmesinde


from sklearn.svm import SVC

# SVM sınıflandırıcı
svm_classifier = SVC(kernel='linear')  # Çekirdek olarak lineer kernel kullandık, farklı çekirdekleri deneyebilirsiniz

# Eğitim
svm_classifier.fit(X_train, y_train)

# Test
y_pred_svm = svm_classifier.predict(X_test)

# Sonuçlar
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM confusion matrix:")
print(cm_svm)






