import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics
from scipy.io import arff

path ='./artificial/'
databrut = arff.loadarff(open(path+"xclara.arff", 'r')) 
datanp = [[x[0], x[1]] for x in databrut[0]]

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

tps1 = time.time()
model = cluster.AgglomerativeClustering(distance_threshold=0, linkage='ward', n_clusters=None)
model = model.fit( datanp )
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.show()
print( " nb clusters = " , k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# set the number of clusters
k = 4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single' , n_clusters = k )
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(datanp, labels))
# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.show()
print( " nb clusters = " , k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )