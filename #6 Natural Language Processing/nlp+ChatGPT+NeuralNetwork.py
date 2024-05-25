import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import confusion_matrix, classification_report

# Veri setini yükleme
yorumlar = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')

# Gerekli NLTK verilerini indirme
nltk.download('stopwords')

# PorterStemmer ve stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Metin ön işleme
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    return text

yorumlar['Processed_Review'] = yorumlar['Review'].apply(preprocess_text)

# Özellik çıkarımı
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(yorumlar['Processed_Review'])
X = tokenizer.texts_to_sequences(yorumlar['Processed_Review'])
X = pad_sequences(X, maxlen=max_len)

y = yorumlar.iloc[:, 1].values

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Modeli oluşturma
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Modeli değerlendirme
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Sonuçlar
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
