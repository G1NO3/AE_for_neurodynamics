import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import Dataset
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

HIDDEN_UNIT = 20
TRAIN_SIZE = 1600
TEST_SIZE = 400
TOTAL_SIZE = TRAIN_SIZE+TEST_SIZE
BATCH_SIZE = 16
train_index = np.arange(TRAIN_SIZE + TEST_SIZE)
np.random.shuffle(train_index)
T = 4001

NAME_full = 'data/fulltraj_for_timesae.pth'

class NeuronTrainSet(Dataset):
    def __init__(self):
        self.data = torch.load(NAME_full)[train_index[:TRAIN_SIZE]]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

class NeuronTestSet(Dataset):
    def __init__(self):
        self.data = torch.load(NAME_full)[train_index[TRAIN_SIZE:]]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


class NeuronSet(Dataset):
    def __init__(self):
        self.data = torch.load(NAME_full)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def FFT_for_Period(x, k=5):
    # [B, T, C]
    xf = torch.fft.rfft(x,dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:,top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            output = self.kernels[i](x)
            res_list.append(output)
        res = torch.stack(res_list, dim=-1)
        return res.mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, configs:Dict):
        super().__init__()
        for k, v in configs.items():
            setattr(self,k,v)
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(self.n_origin_channel, self.n_block_middle,
                               num_kernels=self.incep_n_kernels),
            nn.GELU(),
            Inception_Block_V1(self.n_block_middle, self.n_origin_channel,
                               num_kernels=self.incep_n_kernels)
        )

    def forward(self, x):
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        B, T, C = x.size()
        for i in range(self.k):
            period = period_list[i]
            # padding
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros(B, (length - T), C).to(device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            # reshape
            out = out.reshape(B, length // period, period, C).permute(0,3,1,2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0,2,3,1).reshape(B,-1,C)
            res.append(out[:,:T,:])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation

        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1,T,C,1)
        res = torch.sum(res * period_weight,-1)
        # residual connection
        res = res + x
        return res


class TimesAE(nn.Module):

    def __init__(self, configs:Dict):
        super().__init__()
        for k, v in configs.items():
            setattr(self,k,v)
        
        self.encode_timesblocks = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(self.n_layers)])
        self.encode_projection = nn.Linear(T,HIDDEN_UNIT)

        self.decode_projection = nn.Linear(HIDDEN_UNIT,T)
        self.decode_timesblocks = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(self.n_layers)])
        
        self.layer_norm = nn.LayerNorm(self.n_origin_channel)

    def forward(self, x):
        # Normalization from Non-stationary Transformer
        means = x.mean(1,keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        enc_out = x
        

        # Encoder
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.encode_timesblocks[i](enc_out))
        enc_out = enc_out.permute(0,2,1)
        enc_out = self.encode_projection(enc_out)

        if self.drop_channels:
            for i in self.drop_channels:
                enc_out[:,i,:] = 0


        # Decoder
        dec_out = self.decode_projection(enc_out)
        dec_out = dec_out.permute(0,2,1)
        for i in range(self.n_layers):
            dec_out = self.layer_norm(self.decode_timesblocks[i](dec_out))

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.length, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.length, 1))
        return enc_out, dec_out

def loss_fn(decoded, data):
    criterion = nn.MSELoss()
    return criterion(decoded, data)