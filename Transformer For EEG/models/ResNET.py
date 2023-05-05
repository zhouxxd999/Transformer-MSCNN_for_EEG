import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
from torch.random import set_rng_state
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                            kernel_size=3,stride=stride,padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                            kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                            kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            Y += self.conv3(X)
        return F.relu(Y)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.reshape(x.shape[0],-1)

class GlobalAvgPool2d(nn.Module):
    #全局平均池化可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])


def resnet_block(in_channels,out_channels,num_residuals,first_block=False):
    if first_block:
        #第一个模块的通道数同输入通道数相同
        assert in_channels == out_channels
    
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            #非第一个残差块中的第一个卷积改变通道高和宽的大小和通道数
            blk.append(Residual(in_channels,out_channels,use_1x1conv=True,stride=2))
        else:
            blk.append(Residual(out_channels,out_channels,use_1x1conv=True,stride=1))
    return nn.Sequential(*blk)



def create_resnet(in_channels,num_class,resnet_struct=[3,4,6,3]):
    '''
        resnet_18 : resnet_struct = [2,2,2,2]
        resnet_34 : resnet_struct = [3,4,6,3]
    '''

    net = nn.Sequential(
                    nn.Conv2d(in_channels,32,kernel_size=7,stride=2,padding=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

    ##标准的resnet格式 ,  num_residuals 表明resnet块的个数，可修改层数
    net.add_module('resnet_block1',resnet_block(32,32,num_residuals=resnet_struct[0],first_block=True))
    net.add_module('resnet_block2',resnet_block(32,64,num_residuals=resnet_struct[1],first_block=False))
    net.add_module('resnet_block3',resnet_block(64,128,num_residuals=resnet_struct[2],first_block=False))
    net.add_module('resnet_block4',resnet_block(128,256,num_residuals=resnet_struct[3],first_block=False))
    net.add_module('global_avg_pool',nn.Sequential(GlobalAvgPool2d()))
    net.add_module('fc',nn.Sequential(FlattenLayer(),nn.Linear(256,num_class)))    

    return net


'''
image : 5d-array (trials x freband x height x width)
'''

if __name__ == '__main__':

    net = create_resnet(6,95,[3,4,6,3])
    print(net)


    X = torch.rand(2,6,64,1000)
    for name,layer in net.named_children():
        X = layer(X)
        print(name,'  output shape:\t',X.shape)
