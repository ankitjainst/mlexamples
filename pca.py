import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
X = np.array([[1.2,9],[1.95,21],[3.1,28],[3.9,42],[5.1,49],[5.9,61]], np.float32)
plt.plot(X[:,0],X[:,1],"bo",label="Original Data",linestyle="solid")
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca)
X_origLike = pca.inverse_transform(X_pca)
plt.plot(X_origLike[:,0],X_origLike[:,1],"gd",label="Reconstructed Data",linestyle="dashed")
plt.legend(loc='upper left')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
