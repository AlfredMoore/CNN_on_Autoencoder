import torch
import torch.nn as nn
import time

from torch.utils.tensorboard import SummaryWriter


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(6, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(6, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(6, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(64 * 123 * 4, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 64 * 123 * 4),
            nn.Unflatten(1, (64, 123, 4)),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=(6, 3), stride=(2, 1), padding=(2, 1)),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, (6, 3), (2, 1), (2, 1)),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 3, (6, 3), (2, 1), (2, 1)),
            nn.Tanh()
        )
        #     test
        # self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32,
        #                        kernel_size=(6, 3), stride=(2, 1), padding=(2, 1))
        # self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64,
        #                        kernel_size=(6, 3), stride=(2, 1), padding=(2, 1))
        # self.Conv3 = nn.Conv2d(in_channels=64, out_channels=64,
        #                        kernel_size=(6, 3), stride=(2, 1), padding=(2, 1))
        # self.Flatten1 = nn.Flatten()
        # self.Linear1 = nn.Linear(64 * 123 * 4, 10)
        # #Decoder
        # self.Linear2 = nn.Linear(10, 64 * 123 * 4)
        # self.Unflatten1 = nn.Unflatten(1, (64, 123, 4))
        # self.UnConv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
        #                                   kernel_size=(6, 3), stride=(2, 1), padding=(2, 1))
        # self.UnConv2 = nn.ConvTranspose2d(64, 32, (6, 3), (2, 1), (2,1))
        # self.UnConv1 = nn.ConvTranspose2d(32, 3, (6, 3), (2, 1), (2,1))

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded
        # print(input.shape)
        # data1 = self.Conv1(input)
        # print(data1.shape)
        # data1 = self.Conv2(data1)
        # print(data1.shape)
        # data1 = self.Conv3(data1)
        # print(data1.shape)
        # data1 = self.Flatten1(data1)
        # print(data1.shape)
        # # print(data1)
        # data1 = self.Linear1(data1)
        # print(data1.shape)
        # # print(data2)
        #
        # data2 = self.Linear2(data1)
        # print(data2.shape)
        # data2 = self.Unflatten1(data2)
        # print(data2.shape)
        # data2 = self.UnConv3(data2)
        # print(data2.shape)
        # data2 = self.UnConv2(data2)
        # print(data2.shape)
        # data2 = self.UnConv1(data2)
        # print(data2.shape)
        # return data1,data2


if __name__ == "__main__":
    CAE = ConvAutoEncoder()

    input1 = torch.ones((1, 3, 984, 4))
    print(input1.shape)
    encoded, output1 = CAE(input1)
    print(output1.shape)

    # writer = SummaryWriter("Model")
    # writer.add_graph(CAE, input1)
    # writer.close()
