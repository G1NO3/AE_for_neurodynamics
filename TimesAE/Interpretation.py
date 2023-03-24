from TimesAE import *
import torch
import numpy as np
import matplotlib.pyplot as plt

remain_channels = [0,2,3]
drop_channels = [i for i in range(4) if i not in remain_channels]
model = TimesAE({
    'n_origin_channel':4,
    'n_block_middle':64,
    'incep_n_kernels':3,
    'n_layers':4,
    'length':4001,
    'k':4,
    'drop_channels':drop_channels
}).to(device)
model.load_state_dict(torch.load('TimesAE.pt'))
np.random.seed(2023)
index = np.random.randint(0,TRAIN_SIZE,6)

dataset = NeuronSet()

# model = Ponicare_mapping.DNN()
# model.load_state_dict(torch.load('mapping_2.pt'))
# index = np.random.randint(0,TRAIN_SIZE,10)
# trainset = Ponicare_mapping.TrainSet()
# testset = Ponicare_mapping.TestSet()

#####For visualizing the change of voltage
# model.eval()
# with torch.no_grad():
#     for i in range(len(index)):
#         plt.subplot(3,2,i+1)
#         data = dataset[index[i]:index[i]+1].to(device)
#         encoded, decoded = model(data)
#         x_prime = decoded.detach().cpu().numpy()[0,:,0]
#         print(x_prime.shape)
#         plt.plot(x_prime)

#         x = data.detach().cpu().numpy()[0,:,0]
#         print(x.shape)
#         plt.plot(x)
#         plt.legend(['decoded','data'])
#         loss = loss_fn(decoded, data)
#         print('data:')
#         print(data)
#         print('encoded:',encoded)
#         print('decoded:')
#         print(decoded)
#         print('loss:')
#         print(loss)
#         print()
# plt.suptitle('Remain Channel:' + ' '.join(str(remain_channels)))
# plt.show()


# model.eval()
# with torch.no_grad():
#     for i in index:
#         print(i)
#         pre,post = testset[i:i+1]
#         post_pred = model(pre)
#         loss = test_loss(post_pred, post)
#         print('pre:')
#         print(pre)
#         print('post:',post)
#         print('post_pred:',post_pred)
#         print('loss:')
#         print(loss)
#         print()
# z=torch.tensor(np.random.normal(0,3,HIDDEN_UNIT)).float().reshape(1,-1)
# x_p=model.decoding(z)
# print(x_p)


index = np.arange(70)
model.eval()

with torch.no_grad():
    for i in range(len(index)):
        data = dataset[index[i]:index[i]+1].to(device)
        encoded, decoded = model(data)
        z = encoded.detach().cpu().numpy().squeeze()
        plt.subplot(211)
        plt.plot(z[0])
        plt.subplot(212)
        plt.plot(z[2])    
plt.show()