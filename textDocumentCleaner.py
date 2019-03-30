import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np  
from sklearn.cluster import KMeans 
import matplotlib.colors as mc;

#Read Image
img = mpimg.imread('atharvaveda.png') 
#print(img.shape)

#Show original image
plt.figure("Before processing")
plt.imshow(img) 

#Prepare samples from image
X = np.reshape(img,(-1,4))

#Preprocess the colors to HSV from RGB
hsv = mc.rgb_to_hsv(X[:,0:3])

#Find clusters
myclusters = 3
kmeans = KMeans(n_clusters=myclusters)  
kmeans.fit(hsv)  

#Identify cluster of interest
print(kmeans.cluster_centers_)
clusterCenterBrightness = kmeans.cluster_centers_[:,2]
minBrightCluster = np.argmin(clusterCenterBrightness, 0)
print(minBrightCluster)

#Recolor other clusters to white
for i in range(myclusters) :
    if(i != minBrightCluster) :
        X[kmeans.labels_==i] = [1,1,1,1]

#Reshape samples back to image        
Y = np.reshape(X, (img.shape[0], img.shape[1], -1))

#Show final plot
plt.figure("After processing")
plt.imshow(Y)
plt.show()

