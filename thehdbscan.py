import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
import pandas as pd
import hdbscan
from sklearn import metrics

path ='./artificial/'
#databrut = arff.loadarff(open(path+"dpb.arff", 'r')) 
#datanp = [[x[0], x[1]] for x in databrut[0]]
databrut = pd.read_csv(path+"zz2.txt", sep=" ", encoding="ISO-8859-1", skipinitialspace="True")
datanp = databrut.to_numpy() #[[x[0], x[1]] for x in databrut[0]]

#affichage 2D
#extraction chaque valeur feature pour en faire une liste
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

tps1=time.time()
model = hdbscan.HDBSCAN(min_cluster_size=10) 
model.fit(datanp)
tps2=time.time()
labels = model.labels_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees après hdbscan")
plt.show()
print(metrics.silhouette_score(datanp, labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(datanp, labels))
print(metrics.davies_bouldin_score(datanp, labels))
print(" runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )