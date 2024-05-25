# Random Selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

import random

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


# Tamamen rastgele seçimler yapan öğrenmeyen algoritma


