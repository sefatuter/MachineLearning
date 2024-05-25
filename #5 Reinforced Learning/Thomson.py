import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")


#--> Thomson Sampling

N = 10000 # 10.000 işlem, tıklama, transaction..
d = 10  # ilan sayısı
toplam = 0 # Toplam ödül
secilenler = []

birler = [0] * d
sifirlar = [0] * d


# Her bir ilanın tıklanıp tıklanmadığına bakıcaz, ve tıklandıysa bu değeri döndüreceğiz
for n in range(1,N): # 10000 ilanı dönüyorum
    ad = 0 # Seçmek istediğimiz ilan. Seçilen ilan
    max_th = 0
    for i in range(0,d): # 10 ilandan hangisine tıklayacağımı seçiyorum UCB bularak teker teker
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)    
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
        
    secilenler.append(ad)
    odul = veriler.values[n, ad] # verilerdeki n. satır = 1 ise odul 1
    if odul == 1:
        birler[ad] += 1
    else:
        sifirlar[ad] += 1
    toplam = toplam + odul

print("Toplam Odul: ")
print(toplam)

plt.hist(secilenler)
plt.show()




