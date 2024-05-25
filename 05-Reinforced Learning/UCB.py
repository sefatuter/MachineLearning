import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

#--> Random Selection

'''import random

N = 10000
d = 10 # Tıklanan ilan Ad1, Ad2...
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n, ad] # Verilerdeki n. satır = 1 ise odul 1, (Hangisini rastgele seçtiğimizi gösteriyoruz)    
    toplam = toplam + odul # Tıklanılan değer toplam olarak döner


plt.hist(secilenler)
plt.show()
'''

#--> Upper Confidence Bound (UCB) 
# Üst Güven Sınırı
# Geçmiş tecrübelerden ders alarak giden bir algoritma
# Daha başarılı seçimler yapan algoritma

N = 10000 # 10.000 işlem, tıklama, transaction..
d = 10  # ilan sayısı
# Formülde Ri(n)
oduller = [0] * d # oduller 10 tane elemanlı bir liste olacak ve her eleman 0 olacak her ödül değeri 0 olacak
# Formülde Ni(n)
tiklamalar = [0] * d # O ana kadarki tıklamalar
toplam = 0 # Toplam ödül
secilenler = []
# Her bir ilanın tıklanıp tıklanmadığına bakıcaz, ve tıklandıysa bu değeri döndüreceğiz
for n in range(1,N): # 10000 ilanı dönüyorum
    ad = 0 # Seçmek istediğimiz ilan. Seçilen ilan
    max_ucb = 0
    for i in range(0,d): # 10 ilandan hangisine tıklayacağımı seçiyorum UCB bularak teker teker
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i]) # screenshot'taki formülü yazdık
            ucb = ortalama + delta
        else:
            ucb = N*10
            
        if max_ucb < ucb:  # Max'tan büyük bir ucb çıkarsa max'ı güncelle
            max_ucb = ucb
            ad = i
        # Teker teker ad'lerin ucblerine bakıyorum ve hangisine tıklamam gerektiğini seçiyor, öğrenerek
    
    secilenler.append(ad)
    tiklamalar[ad] += 1
    odul = veriler.values[n, ad] # verilerdeki n. satır
    oduller[ad] += odul
    toplam = toplam + odul

print("Toplam Odul: ")
print(toplam)

plt.hist(secilenler)
plt.show()


