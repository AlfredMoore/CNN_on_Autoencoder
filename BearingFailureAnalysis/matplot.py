import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d    #3d


# extract data
decoded_save_path = "E:\\课程及其实验\\毕业设计\\DataFiles\\DecodedData"

for i in range(4):
    filename = "decoded_bearing_{}.txt".format(i + 1)
    data = np.loadtxt(os.path.join(decoded_save_path, filename))
    (a, b) = data.shape
    data = np.reshape(data, (1, a, b))
    if i == 0:
        decoded_bearing_data = data
    else:
        decoded_bearing_data = np.concatenate((decoded_bearing_data, data), axis=0)
    # decoded_bearing_data.append(data)
    # print(data)
    # decoded_bearing_data = np.concatenate((decoded_bearing_data,data),axis=0)
# print(decoded_bearing_data)
pre_CAE_path = "E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\2nd_test.mat"
pre_CAE_mat = scio.loadmat(pre_CAE_path)
IG_a = pre_CAE_mat.get('IG_a')
IG_b = pre_CAE_mat.get('IG_b')
IG_m = pre_CAE_mat.get('IG_m')
(a, b) = IG_a.shape
IG_a = np.reshape(IG_a, (1, a, b))
IG_b = np.reshape(IG_b, (1, a, b))
IG_m = np.reshape(IG_m, (1, a, b))
pre_CAE_data = np.concatenate((IG_a, IG_b, IG_m), axis=0)
# print(pre_SAE_data.shape)


# plot bearing data
plt.figure(1)
plot_decoded_data = decoded_bearing_data.transpose(0, 2, 1)
plot_pre_CAE_data = pre_CAE_data.transpose(2, 0, 1)
(a, b, c) = plot_decoded_data.shape

plot_step = 20
for j in range(4):
    plt.subplot(2,2,j+1)
    plt.title("Bearing{}".format(j+1))
    for i in range(b):
        plt.scatter(range(len(plot_pre_CAE_data[0][i][::plot_step])), plot_pre_CAE_data[0][i][::plot_step],
                    marker='s',
                    s=10, c='b')
        plt.scatter(range(len(plot_decoded_data[0][i][::plot_step])), plot_decoded_data[0][i][::plot_step],
                    marker='+',
                    s=4, c='r')
plt.suptitle("CAE Training")
# plt.show()

# plot loss_record
plt.figure(2)
loss_save_path = "E:\\课程及其实验\\毕业设计\\DataFiles\\DecodedData\\loss_record.txt"
loss_record = np.loadtxt(loss_save_path)
plt.plot(loss_record)


plt.show()
