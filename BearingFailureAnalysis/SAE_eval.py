from torch.utils.tensorboard import SummaryWriter
from ConvAutoEncoder import *
from DealDataset_mat_2 import *
import torch.cuda
import os
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda")

AE_Model = ConvAutoEncoder()
AE_Model = AE_Model.to(device)

loss_fn = nn.MSELoss()  # MSELoss函数
loss_fn = loss_fn.to(device)

# Decoded_data = []

AE_Model.eval()
print("\n")

save_path_encoded = "E:\\课程及其实验\\毕业设计\\DataFiles\\EncodedData"

with torch.no_grad():
    train_dataset = DealDataset_mat_2(data_path_2="E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\2nd_test.mat")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)

        encoded, decoded = AE_Model(x)
        loss = loss_fn(decoded, y)

    encoded_numpy = encoded.cpu().detach().numpy()
    (ea, eb) = encoded_numpy.shape
    print("Encoded_0:", encoded_numpy)
    np.savetxt(os.path.join(save_path_encoded, "Encoded_0.txt"), encoded_numpy)

with torch.no_grad():
    for i in range(6):
        data_n = i + 1
        train_dataset = DealDataset_mat_2(data_path_2="E:\\课程及其实验\\毕业设计\\Codes\\Matlab\\4th_test_{}.mat".format(data_n))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
        for data in train_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)

            encoded, decoded = AE_Model(x)
            loss = loss_fn(decoded, y)

        encoded_numpy = encoded.cpu().detach().numpy()
        (ea, eb) = encoded_numpy.shape
        print("Encoded_{}:".format(data_n), encoded_numpy)
        np.savetxt(os.path.join(save_path_encoded, "Encoded_{}.txt".format(data_n)), encoded_numpy)

print("\nSave Successfully!")
