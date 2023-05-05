import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time

def get_now_time():
    """
    获取当前日期时间
    :return:当前日期时间
    """
    now =  time.localtime()
    now_time = time.strftime("%Y-%m-%d_%H-%M-%S", now)
    # now_time = time.strftime("%Y-%m-%d ", now)
    return now_time

class Config(object):

    """配置参数"""
    def __init__(self):
        self.device = torch.device('cuda')   # 设备
        self.dropout = 0.2                                              # 随机失活
        self.num_classes = 2                                        # 类别数

        self.num_epochs = 10                                           # epoch数
        self.batch_size = 32                                          # mini-batch大小
        self.learning_rate = 1e-4                                       # 学习率
        self.hidden = 2048                                      
        self.last_hidden = 512

        self.freband_start = [4,8,12,20,32]      #多频段的起始频段
        self.freband_range = 8                   #频段宽度
        self.channels = 64                       #脑电通道数            

        nowTime = get_now_time()
        self.model_name = 'output/MSCNN'+nowTime

class Model(nn.Module):

    def __init__(self,config):
        super(Model, self).__init__()
        self.num_classes = config.num_classes
        self.len_freband = len(config.freband_start)
        self.channels = config.channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,4,(1,5),stride=(1,1),padding=(0,2)),
            nn.ReLU(),
            nn.MaxPool2d((1,15),(1,15))
            )    

        self.conv2 = nn.Sequential(
            nn.Conv2d(1,4,(1,3),stride=(1,1),padding=(0,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
            nn.MaxPool2d((1,15),(1,15)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1,4,(1,1),(1,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
            nn.MaxPool2d((1,15),(1,15)),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d((1,15),(1,15)),
            nn.Conv2d(1,4,(1,1),(1,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
        )

        self.cl = nn.Sequential(
            nn.Conv2d(16,32,(1,1),(1,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(), 
            nn.MaxPool2d((1,15),(1,15)),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.len_freband*32*self.channels*2,config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden,config.last_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.last_hidden,self.num_classes)

        )



    def forward(self, img):
        
        '''
        image : 5d-array (trials x freband x channels x height x width)
        '''
        
        for i in range(self.len_freband):

            #print('input_img shape  :  ',img.shape)
            feature1 = self.conv1(img[:,i,:,:,:])
            feature2 = self.conv2(img[:,i,:,:,:])
            feature3 = self.conv3(img[:,i,:,:,:])
            feature4 = self.conv4(img[:,i,:,:,:])

            feature = torch.concat((feature1,feature2,feature3,feature4),dim=1)

            

            fre1_feat = self.cl(feature)
            if i  == 0:
                feature_conb = fre1_feat
            else: 
                feature_conb = torch.concat((feature_conb,fre1_feat),dim=1)
        
        #print('feature_conb shape : ',feature_conb.shape)
        ''' 
        print('img size : ',img.shape)
        print('freband 1 feature shape: ',fre1_feat.size())
        print('feature shape: ',feature_conb.size())
        '''
        output = self.fc(feature_conb.reshape(img.shape[0],-1))
        return output