Hin = 246
Win = 4
kernel_size = [6, 3]
stride = [2, 1]
padding = [2, 1]
dilation = 1
Hout = (Hin + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1) / stride[0]+1
print('Hout:', Hout)
Wout = (Win + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1]+1
print('Wout:', Wout)
