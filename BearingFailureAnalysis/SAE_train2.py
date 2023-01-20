from ConvAutoEncoder import *
from DealDataset_mat_2 import *
import torch.cuda
import os
import numpy as np
from torch.utils.data import DataLoader

root_path = "E:\\课程及其实验\\毕业设计\\Codes\\Matlab"
data_path = []
for i in range(7):
    if i==0:
        data_path.append(os.path.join(root_path, "2nd_test.mat"))
    else:
        data_path.append(os.path.join(root_path,"4th_test_{}.mat".format(i)))

device = torch.device("cuda")

AE_Model = ConvAutoEncoder()
AE_Model = AE_Model.to(device)

loss_fn = nn.MSELoss()  # MSELoss函数
loss_fn = loss_fn.to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(AE_Model.parameters(), lr=learning_rate, weight_decay=1e-5)

loss_record = []
total_train_step = 0
epoch = 100
out_epoch = 10

AE_Model.train()

for path in data_path:
    print("\n",path)
    train_dataset = DealDataset_mat_2(path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)
    for j in range(out_epoch):
        turn_train_loss = 0
        turn_train_step = 0
        print("-----Turning {} -----".format(j + 1))
        for i in range(epoch):
            for data in train_dataloader:
                x, y = data
                x = x.to(device)
                y = y.to(device)

                encoded, decoded = AE_Model(x)
                loss = loss_fn(decoded, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_step += 1
                turn_train_step += 1
                turn_train_loss += loss.item()
                loss_record.append(loss.item())

                # writer.add_scalar("train loss", loss.item(), total_train_step)
                # print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
        print("本轮训练误差：{},训练次数：{}".format(turn_train_loss, turn_train_step))
# loss_record = np.array(loss_record)
# # print(loss_record)
# print("min loss:",min(loss_record))
#
# # plot
# save_path = "E:\\课程及其实验\\毕业设计\\DataFiles\\DecodedData"
# save_path_encoded = "E:\\课程及其实验\\毕业设计\\DataFiles\\EncodedData"
#
# decoded_numpy = decoded.cpu().detach().numpy()
# encoded_numpy = encoded.cpu().detach().numpy()
# (a, b, c, d) = decoded_numpy.shape
# (ea,eb) = encoded_numpy.shape
#
# decoded_numpy = np.reshape(decoded_numpy, (b, c, d))
# channel = 0
# bearing = 0
# for item in decoded_numpy:
#     filename = "decoded_channel_{}.txt".format(channel + 1)
#     np.savetxt(os.path.join(save_path,filename), item)
#     channel += 1
# for item in decoded_numpy.transpose():
#     filename = "decoded_bearing_{}.txt".format(bearing + 1)
#     np.savetxt(os.path.join(save_path,filename), item)
#     bearing += 1
#
# np.savetxt(os.path.join(save_path,"loss_record1.txt"),loss_record)
# np.savetxt(os.path.join(save_path_encoded,"Encoded_0.txt"),encoded_numpy)

torch.save(AE_Model, "CAE_Model")
print("AE Model Saved Successfully!")
