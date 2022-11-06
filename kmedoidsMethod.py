import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.io import arff

path ='./artificial/'
databrut = arff.loadarff(open(path+"xclara.arff", 'r')) 
datanp = [[x[0], x[1]] for x in databrut[0]]

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

metric1 = -1.0
k=2
while True:
    distmatrix = manhattan_distances(datanp)
    fp = kmedoids.fasterpam(distmatrix,k)
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels

    if metrics.silhouette_score(datanp, labels_kmed, metric='euclidean') > metric1 :
        k=k+1
        metric1= metrics.silhouette_score(datanp, labels_kmed, metric='euclidean')
        print("Loss with FasterPAM:" , fp.loss)
        
    else:
        k = k-1 
        tps1= time.time()
        distmatrix = euclidean_distances(datanp)
        fp = kmedoids.fasterpam(distmatrix,k)
        iter_kmed = fp.n_iter
        labels_kmed = fp.labels
        tps2 = time.time()
        plt.scatter(f0, f1, c=labels_kmed, s=8)
        plt.title("Donnes apres clustering KMedoids")
        plt.show()
        print("nb clusters = ", k, ", nb iter = ", iter_kmed, ", runtime (ms)= ", round((tps2 - tps1)*1000, 2))
        break
