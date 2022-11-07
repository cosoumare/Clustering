#IMPORT et tests clustering agglomératif pour les datasets mystere

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
filename = "zz2.txt"
databrut = pd.read_csv(path+filename, sep=" ", encoding="ISO-8859-1", skipinitialspace=True)
data = databrut
datanp = databrut.to_numpy()

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]


# set the number of clusters
k = 8
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'ward' , n_clusters = k )
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(datanp, labels))
# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Données après clustering agglomératif - ward " )
plt.show()
print( " nb clusters = " , k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
print(metrics.davies_bouldin_score(datanp, labels))
print(metrics.calinski_harabasz_score(datanp, labels))