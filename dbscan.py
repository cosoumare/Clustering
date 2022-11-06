import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics
from sklearn import neighbors
from scipy.io import arff

path ='./artificial/'
databrut = arff.loadarff(open(path+"smile3.arff", 'r')) 
datanp = [[x[0], x[1]] for x in databrut[0]]

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]


# Distances k plus proches voisins
# Donnees dans X

metric1=-1.0
k=2
while True:
    neigh = neighbors.NearestNeighbors(n_neighbors = k )
    neigh.fit(datanp)
    distances , indices = neigh.kneighbors(datanp)
    # retirer le point " origine "
    newDistances = np.asarray( [ np.average( distances [ i ] [ 1 : ] ) for i in range (0 ,
    distances.shape [ 0 ] ) ] )
    trie = np.sort( newDistances )

    model = cluster.dbscan(eps=trie[-1])
    model.fit(datanp)
    labels = model.labels
    
    
    if metrics.silhouette_score(datanp, labels, metric='euclidean') > metric1 :
        metric1=metrics.silhouette_score(datanp, labels, metric='euclidean')
        k=k+1
    else :
     k=k-1 
     neigh = neighbors.NearestNeighbors(n_neighbors = k )
     neigh.fit(datanp)
     distances , indices = neigh.kneighbors(datanp)
     # retirer le point " origine "
     newDistances = np.asarray( [ np.average( distances [ i ] [ 1 : ] ) for i in range (0 ,
     distances.shape [ 0 ] ) ] )
     trie = np.sort( newDistances )
     tps1=time.time()
     model = cluster.dbscan(eps=trie[-1])
     model.fit(datanp)
     tps2=time.time()
     labels = model.labels
     plt.scatter(f0, f1, c=labels, s=8)
     plt.title("Donnees après dbscan itératif")
     plt.show()
     break

