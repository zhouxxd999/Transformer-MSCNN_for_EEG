import numpy as np
import pandas as pd
from importlib import import_module
import torch
from utils import concatenateAllData,getAllSubClassData,MyDataset
from torch.utils.data import DataLoader
from train import train



model_name = 's-tCNN_Transformer'

data_dir = 'data/train'

x = import_module('models.' + model_name)
config = x.Config()
print('all class number : ',config.num_classes)



base_dir = 'data/train'
from sklearn.model_selection import train_test_split
left_data,right_data,_,_ =concatenateAllData(data_dir,num_sub=10)
leftdata,rightdata,_,_= getAllSubClassData(left_data,right_data,_,_)
all_data = np.concatenate((leftdata,rightdata),axis=0)
label = np.concatenate((np.zeros(leftdata.shape[0],),np.ones(rightdata.shape[0],)))

print(all_data.shape)
print(label.shape)

#切分数据集
train_data,val_data,train_label,val_label = train_test_split(all_data,label,test_size=0.2,random_state=42)


train_dataset = MyDataset(train_data,train_label)
train_loader = DataLoader(train_dataset,config.batch_size,shuffle=True)


val_dataset = MyDataset(val_data,val_label)
val_loader = DataLoader(val_dataset,config.batch_size,shuffle=True)

model = x.Model(config).to(config.device)

train(config,model,train_loader,val_loader,val_loader)




