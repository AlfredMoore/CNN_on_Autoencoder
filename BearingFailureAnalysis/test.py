import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
# from sklearn.datasets.samples_generator import make_blobs

# X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
#                   cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
save_path_encoded = "E:\\课程及其实验\\毕业设计\\DataFiles\\EncodedData"
for i in range(7):
    filename = "Encoded_{}.txt".format(i)
    data = np.loadtxt(os.path.join(save_path_encoded, filename))
    data_shape = np.size(data)
    # print(data.shape)
    data = np.reshape(data, (1, data_shape))
    if i == 0:
        encoded_data = data
    else:
        encoded_data = np.concatenate((encoded_data, data), axis=0)
# print(encoded_data)
#
X = encoded_data
pca = PCA(n_components=3)
pca.fit(X)
# featuers
print(pca.explained_variance_)
print(pca.components_)
# Dimension Reduction
X_new = pca.transform(X)
print(X_new)
fig = plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
plt.show()