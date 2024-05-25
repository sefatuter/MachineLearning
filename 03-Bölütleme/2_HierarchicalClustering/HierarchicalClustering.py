#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

#---> K-Means 

from sklearn.cluster import KMeans
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
plt.show()

kmeans = KMeans(n_clusters = 4, init= 'k-means++', random_state=123)
Y_tahmin = kmeans.fit_predict(X)

plt.scatter(X[Y_tahmin == 0, 0], X[Y_tahmin == 0, 1], s = 100, c = 'red') # 1.cluster
plt.scatter(X[Y_tahmin == 1, 0], X[Y_tahmin == 1, 1], s = 100, c = 'blue') # 2. cluster
plt.scatter(X[Y_tahmin == 2, 0], X[Y_tahmin == 2, 1], s = 100, c = 'green') # 3. cluster
plt.scatter(X[Y_tahmin == 3, 0], X[Y_tahmin == 3, 1], s = 100, c = 'yellow') # 4. cluster
plt.title('--KMeans--')
plt.show()


#---> Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='ward')

Y_tahmin = ac.fit_predict(X)

print(Y_tahmin)

plt.scatter(X[Y_tahmin == 0, 0], X[Y_tahmin == 0, 1], s = 100, c = 'red') # 1.cluster
plt.scatter(X[Y_tahmin == 1, 0], X[Y_tahmin == 1, 1], s = 100, c = 'blue') # 2. cluster
plt.scatter(X[Y_tahmin == 2, 0], X[Y_tahmin == 2, 1], s = 100, c = 'green') # 3. cluster
plt.scatter(X[Y_tahmin == 3, 0], X[Y_tahmin == 3, 1], s = 100, c = 'yellow') # 4. cluster
plt.title('--Hiearchical Clustering--')
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()











