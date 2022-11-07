#import et tests pour les datasets mystere

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

print("Appel KMeans pour une valeur fixe de k")

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

#print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
#print(metrics.davies_bouldin_score(datanp, labels))
#print(metrics.calinski_harabasz_score(datanp, labels))

metric1=-1.0
k=2
while True:
    model=cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    labels=model.labels_
    iteration=model.n_iter_

    
    if metrics.silhouette_score(datanp, labels, metric='euclidean') > metric1 :
        metric1=metrics.silhouette_score(datanp, labels, metric='euclidean')
        k=k+1
    else :
        k=k-1
        tps1=time.time()
        model=cluster.KMeans(n_clusters=k, init='k-means++')
        model.fit(datanp)
        tps2=time.time()
        labels=model.labels_
        iteration=model.n_iter_
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Donnees après clustering Kmeans")
        plt.show()
        print("nb clusters :", k, "; nb iter :", iteration, "; runtime (ms) :", round((tps2-tps1)*1000,2))
        print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
        print(metrics.davies_bouldin_score(datanp, labels))
        print(metrics.calinski_harabasz_score(datanp, labels))
        break