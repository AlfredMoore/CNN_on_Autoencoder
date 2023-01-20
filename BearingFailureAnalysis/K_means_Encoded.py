import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

SS = []
CHS = []
DBS = []

for i in range(4):
    n_clusters = 2 + i
    kmean_encoded = KMeans(n_clusters=n_clusters)
    kmean_encoded.fit(encoded_data)
    result = kmean_encoded.predict(encoded_data)
    centroids = kmean_encoded.cluster_centers_
    # print("K-Means centroids:",centroids)
    print("{}_clusters:".format(2 + i), result)

    # Metrics
    s1 = silhouette_score(encoded_data, result)  # 轮廓系数接近1为合理
    s2 = calinski_harabasz_score(encoded_data, result)  # CH分数越高约合理
    s3 = davies_bouldin_score(encoded_data, result)  # 戴维森堡丁指数接近0为合理

    SS.append(s1)
    CHS.append(s2)
    DBS.append(s3)

print("silhouette_score:", SS)  # 轮廓系数接近1为合理
print("calinski_harabasz_score:", CHS)  # CH分数越高约合理
print("davies_bouldin_score:", DBS)  # 戴维森堡丁指数接近0为合理
# print(result)

# plot
a = 1
b = 2
plt.figure(1)
plt.subplot(a, b, 1)
plt.plot(range(2, 6), SS)
plt.xlabel("K")
plt.title("Silhouette Score")
plt.subplot(a, b, 2)
plt.plot(range(2, 6), CHS)
plt.xlabel("K")
plt.title("Calinski Harabasz Score")
# plt.subplot(a, b, 3)
plt.figure(2)
plt.subplot(a,b,1)
plt.plot(range(2, 6), DBS)
plt.xlabel("K")
plt.title("Davies Bouldin Score")
plt.show()
