import scipy.io as scio
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable


# def ExtractData(data_path, array1, array2):
#     data = scio.loadmat(data_path)
#     data1 = data.get(array1)
#     data2 = data.get(array2)
#     pydata1 = []
#     pydata2 = []
#     for item in data1:
#         for x in item:
#             pydata1.append(x)
#     for item in data2:
#         for x in item:
#             pydata2.append(x)
#     return pydata1, pydata2
#
#
# def ExtractData(data_path):
#     data_mat = scio.loadmat(data_path)
#     tensor_trans = transforms.ToTensor()
#     IG_a = data_mat.get('IG_a')
#     IG_b = data_mat.get('IG_b')
#     IG_m = data_mat.get('IG_m')
#     Tensor_a = tensor_trans(IG_a)
#     Tensor_b = tensor_trans(IG_b)
#     Tensor_m = tensor_trans(IG_m)
#     tensor_data = torch.cat((Tensor_a, Tensor_b, Tensor_m), 0)
#     # tensor_data = torch.reshape(tensor_data, (1, 3, -1, 4))
#     return tensor_data.type(torch.float32)
#
#
# # test
# if __name__ == '__main__':
#     data_path = "E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\2nd_test.mat"
#     tensor_data = ExtractData(data_path)
#     # tensor_data = tensor_data.type(torch.float32)
#     # print(tensor_data.type())
#     print(tensor_data.shape)
#     train_data = ExtractData("E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\2nd_test.mat")
#     print(train_data)
#     train_dataloader = DataLoader(train_data, 64)
#     for data in train_dataloader:
#         imgs, targets = data
#         print(imgs.shape)
#         print(targets.shape)
