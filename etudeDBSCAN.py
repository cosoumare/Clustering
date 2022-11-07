#import et tests méthode dbscan pour les datasets mystere
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from scipy.io import arff
import pandas as pd

path = './datasetRapport/'
filename = "zz1.txt"
databrut = pd.read_csv(path+filename, sep=" ", encoding="ISO-8859-1", skipinitialspace=True)
data = databrut
datanp = databrut.to_numpy()

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
tps1 = time.time()
model = cluster.DBSCAN(eps=0.01661e6, min_samples=5, leaf_size=30)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees après dbscan itératif")
plt.show()
#print( " nb clusters = " , k a " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
print( " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
print(metrics.davies_bouldin_score(datanp, labels))
print(metrics.calinski_harabasz_score(datanp, labels))
