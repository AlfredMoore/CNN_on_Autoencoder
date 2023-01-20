from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import scipy.io as scio
from torchvision import transforms


class DealDataset_mat_2(Dataset):
    def __init__(self,data_path_2):
        # self.data_path = "E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\4th_test.mat"
        self.data_path = data_path_2
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

    train_dataset = DealDataset_mat_2("E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\4th_test.mat")

