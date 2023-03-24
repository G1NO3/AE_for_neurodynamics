import numpy as np
import torch
import matplotlib.pyplot as plt


data = torch.load('data/neurondata_trajectory.pth')
xf = torch.fft.rfft(data,dim=0)
frequency = abs(xf).mean(1)
plt.subplot(211)
index = np.random.randint(0,44,44)
for i in np.arange(5):
    plt.plot(data[:,i],linewidth=1)
plt.legend(np.arange(5))
plt.subplot(212)
plt.plot(frequency)
plt.show()

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf)[:, top_list]