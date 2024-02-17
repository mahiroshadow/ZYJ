# from module.model2 import CRNN
import torch


def train():
    a=torch.Tensor(6,2,3,28,28)
    print(a[:,0,:,:,:].shape)

if __name__=='__mian__':
    train()