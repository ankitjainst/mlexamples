import numpy as np  
from sklearn.cluster import KMeans  

dataToCluster  = [1, 100, 20, 120, 5]
dataToClusterAsNDArray = np.asarray(dataToCluster)
print("The shape of input data is", dataToClusterAsNDArray.shape)

dataToClusterAsNDArray = np.reshape(dataToClusterAsNDArray,(dataToClusterAsNDArray.shape[0],1))
print("Input data is as follows")
print(dataToClusterAsNDArray)

clusterCount = 2
kmeans = KMeans(n_clusters=clusterCount) 

kmeans.fit(dataToClusterAsNDArray)  
print("The labels obtained are", kmeans.labels_)

for clusterNumber in range(clusterCount):
    clusterLogicalIndices = kmeans.labels_== clusterNumber
    print("The logical indices for", clusterNumber ,"are as follows")
    print(clusterLogicalIndices)

    print("The values belonging to the clusterNumber ", clusterNumber, "are as follows")
    print(dataToClusterAsNDArray[clusterLogicalIndices])
