# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None)

tList = []
for i in range (0,7501):
    tList.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori
rules = apriori(tList,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)
# min_support => %1 destek, min_confidence => %2
rules = list(rules)

for item in rules:
    print(item)
    print("\n")