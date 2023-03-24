import numpy as np
import torch
# import matplotlib.pyplot as plt


NAME = 'data/fulltraj_for_timesae.pth'
N_neuron = 400
np.set_printoptions(threshold=np.inf)
array = []
for i in range(1,6):
    f1 = open('FullTrajectory'+str(i)+'.txt')
    datalist = []
    while True:
        a = list(map(int, f1.readline().split()))
        if a :
            datalist.append(a)
        else:
            f1.close()
            break
    b = np.array(datalist).reshape(-1,4,N_neuron)
    array.append(b)
    print(b.shape)
array = np.concatenate(array,axis=-1)



x = torch.tensor(array).permute(2,0,1).float()
print(x[np.random.randint(0,x.shape[0],5)])
print(x.shape)
torch.save(x, NAME)



# N_neuron = 4000
# np.set_printoptions(threshold=np.inf)
# datalist = []
# while True:
#     a = list(map(int, f1.readline().split()))
#     if a :
#         datalist.append(a)
#     else:
#         f1.close()
#         break
# x = np.array(datalist).reshape(-1,N_neuron,4)
# x = torch.tensor(x).permute(1,0,2).float()
# print(x[np.random.randint(0,x.shape[0],5)])
# print(x.shape)
# torch.save(x, NAME)

# datalist = []
# while True:
#     a = list(map(int, f2.readline().split()))[:N_INPUT]
#     if a :
#         datalist.append(a)
#     else:
#         f2.close()
#         break
# x = np.array(datalist)
# print(x.shape)
# print(x[np.random.randint(0,80000,5)])
# torch.save(torch.tensor(np.array(datalist)).float(), NAME_postset)