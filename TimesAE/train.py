import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from TimesAE import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
trainset = NeuronTrainSet()
testset = NeuronTestSet()

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = False)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False)

model = TimesAE({
    'n_origin_channel':4,
    'n_block_middle':64,
    'incep_n_kernels':3,
    'n_layers':4,
    'length':4001,
    'k':4
}).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# print('-----Train-----')
# model.train()
# for epoch in range(20):
#     total_loss = 0.0
#     for x in tqdm(trainloader):
        
#         mu, lnsigma, encoded, decoded = model(x)
#         loss = loss_fn(mu, lnsigma, decoded, x)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss
#     avg_loss = total_loss/(TRAIN_SIZE/BATCH_SIZE)
#     print(avg_loss.item())

# torch.save(model.state_dict(), 'VAE.pt')

# print('-----Test-----')
# model.eval()
# total_loss = 0.0
# with torch.no_grad():
#     for x in tqdm(testloader):
#         mu, lnsigma, encoded, decoded = model(x)
#         loss = test_loss(decoded, x)
#         total_loss += loss
#     avg_loss = total_loss/(TEST_SIZE/BATCH_SIZE)
#     print(avg_loss.item())


model.load_state_dict(torch.load('TimesAE.pt'))
print('-----Train-----')
model.train()
for epoch in range(60):
    total_loss = 0.0
    torch.cuda.empty_cache()
    for x in tqdm(trainloader):
        x = x.to(device)
        encoded, decoded = model(x)
        loss = loss_fn(decoded, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    avg_loss = total_loss/(TRAIN_SIZE/BATCH_SIZE)
    print(avg_loss.item())

torch.save(model.state_dict(), 'TimesAE.pt')

print('-----Test-----')
model.eval()
total_loss = 0.0
with torch.no_grad():
    for x in tqdm(testloader):
        x = x.to(device)
        encoded, decoded = model(x)
        loss = loss_fn(decoded, x)
        total_loss += loss
    avg_loss = total_loss/(TEST_SIZE/BATCH_SIZE)
    print(avg_loss.item())