#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

veriler = pd.read_csv('musteriler.csv')


X = veriler.iloc[:,3:].values # Hacim ve maaş

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X) # Eğittik

print(kmeans.cluster_centers_) # merkezlerini nerede oluşturdu, 3 merkez oluşturdu

sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init= 'k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)


plt.plot(range(1,11), sonuclar) # 4 güzel bir nokta hızlı düşüşten sonra düzlemiş












