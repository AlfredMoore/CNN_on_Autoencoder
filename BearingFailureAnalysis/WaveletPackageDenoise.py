from ExtractData import ExtractData
import pywt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def wavelet_packet_three_level(x):
    mother_wavelet = 'db8'
    wp = pywt.WaveletPacket(data=x, wavelet=mother_wavelet, mode='symmetric', maxlevel=3)
    node_name_list = [node.path for node in wp.get_level(3, 'natural')]
    rec_results = []
    for i in node_name_list:
        new_wp = pywt.WaveletPacket(data=np.zeros(len(x)), wavelet=mother_wavelet, mode='symmetric')
        new_wp[i] = wp[i].data
        x_i = new_wp.reconstruct(update=True)
        rec_results.append(x_i)
    output = np.array(rec_results)
    return output

def plot_file(data,pdf_name):
    Y = data
    X = np.linspace(0, len(Y) / 12000, len(Y))
    fig, ax = plt.subplots()
    ax.plot(X, Y, color='C1')
    fig.savefig(pdf_name)

data_path="E:\\课程及其实验\\毕业设计\\DataFiles\\NormalBaseline\\MotorLoad0HP_MotorSpeed1797rpm.mat"
array1='X097_DE_time'
array2='X097_FE_time'
data = ExtractData(data_path,array1,array2)
datarec1=wavelet_packet_three_level(data[0])
datarec2=wavelet_packet_three_level(data[1])

plot_file(data[0],"BeforeWpt.jpg")
plot_file(datarec1.data,"AfterWpt.jpg")


# w=pywt.Wavelet('db8')
# # l=pywt.dwt_max_level(len(data),w.dec_len)
# # print("max level=",l)
# l=3
# coeffs = pywt.wavedec2(data,wavelet='db8',level=l)
# threshold=0.04
# for i in range(1,len(coeffs)):
#     coeffs[i]=pywt.threshold(coeffs[i],threshold*max(coeffs[i]))
# datarec=pywt.waverec2(coeffs,'db8')
# mintime=0
# maxtime=mintime+len(data)+1