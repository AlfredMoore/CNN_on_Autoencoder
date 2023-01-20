from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import scipy.io as scio
from torchvision import transforms


class DealDataset_mat(Dataset):
    def __init__(self):
        self.data_path = "E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\2nd_test.mat"
        data_mat = scio.loadmat(self.data_path)
        tensor_trans = transforms.ToTensor()
        IG_a = data_mat.get('IG_a')
        IG_b = data_mat.get('IG_b')
        IG_m = data_mat.get('IG_m')
        Tensor_a = tensor_trans(IG_a)
        Tensor_b = tensor_trans(IG_b)
        Tensor_m = tensor_trans(IG_m)
        self.IG_x = (torch.cat((Tensor_a, Tensor_b, Tensor_m), 0)).type(torch.float32)
        self.IG_x = torch.reshape(self.IG_x, (1, 3, -1, 4))
        self.IG_y = self.IG_x
        self.len = self.IG_x.shape[0]

    def __getitem__(self, item):
        return self.IG_x[item], self.IG_y[item]

    def __len__(self):
        return self.len


if __name__ == "__main__":

    train_dataset = DealDataset_mat()
    train_loader = DataLoader(dataset=train_dataset, batch_size=1)
    for i, data in enumerate(train_loader):
        x, y = data
        print(i, "input", x.data.size(), "labels", y.data.size())

    # data_path = "E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\2nd_test.mat"
    # data_mat = scio.loadmat(data_path)
    # tensor_trans = transforms.ToTensor()
    # IG_a = data_mat.get('IG_a')
    # IG_b = data_mat.get('IG_b')
    # IG_m = data_mat.get('IG_m')
    # Tensor_a = tensor_trans(IG_a)
    # Tensor_b = tensor_trans(IG_b)
    # Tensor_m = tensor_trans(IG_m)
    # IG_x = torch.cat((Tensor_a, Tensor_b, Tensor_m), 0)
    # IG_x = torch.reshape(IG_x, (1, 3, -1, 4))
    # IG_y = IG_x
    # # print(IG_x.size())
    # deal_dataset = TensorDataset(IG_x, IG_y)
    # train_loader = DataLoader(dataset=deal_dataset,
    #                           batch_size=1,
    #                           shuffle=True,
    #                           num_workers=0)
    #
    # for i, data in enumerate(train_loader):
    #     inputs, labels = data
    #     # inputs, labels = Variable(inputs), Variable(labels)
    #     print(i, "input", inputs.data.size(), "labels", labels.data.size())
