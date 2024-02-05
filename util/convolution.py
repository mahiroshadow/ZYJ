import torch
import numpy as np
import torch.nn as nn
T=tuple|torch.Tensor|np.ndarray

'''
可以对Conv2D/MaxPool2D输出的(h,w)进行计算
'''
# 卷积计算器(计算输出width/height)
def convolutionCompute(w:T,kernel_size:T,padding:T,stride:T):
    if not isinstance(w,torch.Tensor) or not isinstance(kernel_size,torch.Tensor) or not isinstance(padding,torch.Tensor) or not isinstance(stride,torch.Tensor):
        w,kernel_size,padding,stride=torch.Tensor(w),torch.Tensor(kernel_size),torch.Tensor(padding),torch.Tensor(stride)
    output=(w-kernel_size+2*padding)//stride+1
    return output

if __name__=='__main__':
    # output=convolutionCompute((28,28),(3,3),(0,0),(2,2))
    # print(output)
    a=torch.FloatTensor([[1.0,2.0],[2.0,3.0]])
    bn=nn.BatchNorm1d(2)
    linear=nn.Linear(2,16)
    print(bn(a))