import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy import signal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset



def psd(trials,NFFT,Fs):
    '''
    计算每一个trials的PSD功率谱密度
    
    Parameters
    ————————————————
    trials : 3d-array (trials x channels x samples)
        the EEG signal
    
    Returns
    ______________
    trial_PSD : 3d-array (trials x channels x PSD)
        the PSD for each trials
    freqs : list of floats
        the frequency for which the psd was computed (useful for plotting later)  

    '''
    ntrials = trials.shape[0]
    nchannels = trials.shape[1]

    #尝试计算一个PSD，查看PSD的数据长度
    (a,b) = mlab.psd(trials[0,0,:],NFFT=NFFT,Fs=Fs)
    npsdsamples = len(b)

    trials_psd = np.zeros((ntrials,nchannels,npsdsamples))

    #iterate over trials ans channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            #caculate the psd 
            (freqs,PSD) = signal.welch(trials[trial,ch,:],nperseg = NFFT,fs=Fs)
            trials_psd[trial,ch,:] = PSD.ravel()

    return trials_psd,freqs


def plot_psd(PSD_data_c1,PSD_data_c2,freqs,ch_names,ch_idx,freq_range=[0,30]):
    '''
    画出3个选中信号的PSD图像  1x3排列
    
    Parameters
    ——————————————————————————————————————


    '''

    plt.figure(figsize=(12,5))
    for i in range(len(ch_names)):
        ax = plt.subplot(1, 3, i+1, frameon = False)
        plt.plot(freqs,np.mean(PSD_data_c1[:,ch_idx[i],:],axis=0),label='c1')
        plt.plot(freqs,np.mean(PSD_data_c2[:,ch_idx[i],:],axis=0),label='c2')
        plt.title(ch_names[i]+'  psd')
        plt.xlim(freq_range[0],freq_range[1])
       
    plt.show()




def band_pass(trials,lo,hi,sample_rate):
    '''
    为信号设计及使用带通滤波器

    Parameters
    ————————————————————————
    trials : 3d-array (trials x channels x samples)
        the eeg signal
    lo : float
        Lower frequency bound (in hz)
    hi : float
        Upper frequency bound (in hz)
    sample_rate : float
        Sample rate of the signal (in hz)

    Returns:
    ————————————————————————
    trials_filt : 3d-array (trials x channels x samples)
        The bandpass signal
    '''

    a , b = signal.iirfilter(6,[lo/(sample_rate/2.0) , hi/(sample_rate/2.0)]) 
    
    #为每一个trial的数据应用该滤波器
    ntrials = trials.shape[0]
    trials_filter = np.zeros(trials.shape)
    print('trials_filter shape ',trials_filter.shape)

    for i in range(ntrials):
        trials_filter[i,:,:] = signal.filtfilt(a,b,trials[i,:,:],axis=1)

    return trials_filter


def logvar(trials):
    '''
    计算每一个trial的方差variance

    Parameters
    ————————————————————————————————————————
    trials : 3d-array (trials x channels x samples)
        trial的数据
    '''

    return np.log(np.var(trials,axis=2))

def plot_logvar(left_trials,right_trials):
    '''
    画出每一个通道的log-var成分 
    '''

    plt.figure(figsize = (12,5))
    
    nchannels = left_trials.shape[1]

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.4
    print(x0.shape) 
    y0 = np.mean(left_trials,axis=0)
    y1 = np.mean(right_trials,axis=0)

    print(y0.shape)

    plt.bar(x0,y0,width=0.3,color='b',label='c1')
    plt.bar(x1,y1,width=0.25,color='r',label='c2')
    
    plt.xlim(-0.5,nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel / compoment')
    plt.xlabel('channels / compoment')
    plt.ylabel('log-var')
    plt.legend()





def cal_norm_covMatrix(X):
    '''
    计算X输入的协方差矩阵

    Parameters
    ——————————————————————————
    X : 2d-array  (channels x samples)
        一个trial的数据

    Return
    ——————————————————————————
    R_norm : 2d-array (channels x channels)
        一个trial的协方差矩阵
    '''
    R = X.dot(X.T)
    R_norm = R / X.shape[1]
    return R_norm


def cal_whitening_P(Rc):
    '''
    计算CSP的白化矩阵P   P = 1/sqrt(lambda) * U

    Parameters
    ——————————————————————————
    X : 2d-array  (channels x channels)
        Rc = Ra_avg + Rb_avg   
           Ra_avg 为某一类数据的平均协方差矩阵
           Rb_abg 为另一类数据的平均协方差矩阵

    Return
    ——————————————————————————
    P : 2d-array (channels x channels)
        P = 1/sqrt(lambda) * U  白化矩阵
    '''
    U,lambd,_ = np.linalg.svd(Rc)
    #print(lambd)
    #print('U特征向量矩阵为:\n',U)
    
    #print('RC value is :',Rc[0])
    #print('reverse Rc value is :',U.dot(np.diag(lambd).dot(U.T))[0])

    P = np.diag(lambd**(-0.5)).dot(U.T)
    #print('定义矩阵P为:\n',P)
    return P

def cal_W(left_trials,right_trials):
    '''
    计算CSP的投影矩阵

    Parameters
    ——————————————————————————
    left_trials : 3d-array  (trials x channels x samples)
        左运动想象数据
    right_trials : 3d-array  (trials x channels x samples)
        右运动想象数据   

    Return
    ——————————————————————————
    W : 2d-array (channels x channels)
        CSP空间投影矩阵W
    '''
    ntrials = left_trials.shape[0]
    nchannels = left_trials.shape[1]

    Ra = np.zeros((ntrials,nchannels,nchannels))
    for i in range(ntrials):
        Ra[i,:,:] = cal_norm_covMatrix(left_trials[i,:,:])
    Ra_avg = np.mean(Ra,axis=0)
    #print('Ra_avg',Ra_avg[0])

    Rb = np.zeros((ntrials,nchannels,nchannels))
    for i in range(ntrials):
        Rb[i,:,:] = cal_norm_covMatrix(right_trials[i,:,:])
    Rb_avg = np.mean(Rb,axis=0)

    Rc = Ra_avg + Rb_avg

    P = cal_whitening_P(Rc)


    Sa = P.dot(Ra_avg.dot(P.T))
    Sb = P.dot(Rb_avg.dot(P.T))

    D1,lambd_a,_ = np.linalg.svd(Sa)
    #D2,lambd_b,_ = np.linalg.svd(Sb)
    print('两类的特征值lambda:\n',lambd_a)
    #print('两类的特征值lambdb:\n',lambd_b)

    #print('Sa value is ',Sa[0])
    #reverse_sa = D1.dot(np.diag(lambd_a).dot(D1.T))
    #print('reverse Sa value is ',reverse_sa[0])
    
    #print('D1 value is \n',D1[0:5,[0,1]])
    #print('D2 value is \n',D2[0:5,[-1,-2]])

    #print('Sb value is ',Sb[0])
    #print('Sb * u ',Sb.dot(D1[:,0]))
    #print('lambd * u ',lambd_b[-1] * D1[:,0])

    print('Sb value is ',Sb[0:6,0:6])
    reverse_sb = D1.dot(np.diag(1-lambd_a).dot(D1.T))
    print('reverse Sb value is ',reverse_sb[0:6,0:6])

    print('------------------------')
    #取第一行和最后一行
    W = D1.T.dot(P)

    return W


def apply_mix(W,trials):
    '''
    对新的数据使用W进行投影

    Parameters
    ——————————————————————————
    W : 2d-array (channels x channels)
        CSP的投影矩阵
    trials : 3d-array (trials x channels x samples)
        需要投影的数据

    Return
    ——————————————————————————
    trials_csp : 3d-arrray(trials x channels x samples)
        使用CSP投影后的数据
    '''    
    ntrials = trials.shape[0]
    trials_csp = np.zeros(trials.shape)
    for i in range(ntrials):
        trials_csp[i,:,:] = W.dot(trials[i,:,:])
    return trials_csp



def cal_csp_feature(csp_var,fea_num=[0,1,-2,-1]):
    '''
    计算CSP投影后数据的方差,并将该方差归一化,以此作为CSP特征
    
    Parameters
    ——————————————————————————
    csp_var : 2d-array (trials x channels)
        数据经过CSP投影过后的方差矩阵
    fea_num : list 
        选择需要的通道作为特征  如[0,-1] 代表选择第一个和最后一个方差特征

    Return
    ——————————————————————————————————————————
    csp_fea : 2d-array (trials x csp feature dim)
        CSP投影未经归一化后的选择方差矩阵
    csp_norm : 2d-array (trials x csp feature dim)
        CSP投影经归一化后的选择方差矩阵
    '''
    csp_fea = abs(csp_var[:,fea_num])
    csp_fea_norm =  np.log(csp_fea / np.sum(csp_fea,axis=1,keepdims=True))
    return csp_fea,csp_fea_norm



import random

def random_list(num_range=13,sample_num=240):
    rand_list = []
    for i in range(num_range):
        rand_list.append( random.randint(0,sample_num-1))
    #print(rand_list)
    return rand_list


def data_augmentation(data_in,multi_times,segment_range=15,handclass='left_hand'):
    '''
        数据增强 _ 分段随机组合

    Parameters
    ——————————————————————————————————
    data_in : 3d-array  (trials x channels x samples)
        3维数据输入
    multi_times : float
        数据增强倍数
    segment_range : float
        一个trial信号切割片段个数   
    handclass : str
        左右手类型  影响输出标签

    Return
    ——————————————————————————————————
    data_aug : 3d-array (augment trials x channels x samples)
        原始数据以及增强后的数据输出
    
    label_aug : 1d-array
        增强数据对应的标签
    '''


    data_aug_count = int((multi_times+1)*data_in.shape[0])
    #print('数据增强的个数：',data_aug_count)
    #print('data in shape:',data_in.shape)
    segment_points = int(data_in.shape[2] / segment_range)
    data_aug = np.zeros((data_aug_count,data_in.shape[1],data_in.shape[2]))
    for i in range(data_aug_count):
       
        rand_list = random_list(num_range=segment_range,sample_num=data_in.shape[0])
        for k in range(segment_range):
            start_seg = k * segment_points
            end_seg = start_seg + segment_points
            data_aug[i,:,start_seg:end_seg] = data_in[rand_list[k],:,start_seg:end_seg]

    data_aug[int((multi_times)*data_in.shape[0]):data_aug_count,:,:] = data_in

    if handclass == 'left_hand':
        label_aug = np.zeros((data_aug_count,))
    else:
        label_aug = np.ones((data_aug_count,))

    #2s滑动窗口
    '''
    data_aug2 = np.zeros((data_aug_count*4,data_in.shape[1],data_in.shape[1],250))
    for i in range(data_aug_count):
        for j in range(4):
            start_se = j*250
            end_se = start_se+250
            data_aug2[4*i+j,:,:] =  data_aug[i,:,start_se:end_se]
    '''
    return data_aug,label_aug





def evaluate_accuracy_gpu(net,test_dataloader,
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """使用GPU计算模型再数据集上面的精度"""
    acc_sum , n = 0.0,0
    with torch.no_grad():
        for i,data in enumerate(test_dataloader):
            X = data[0]
            y = data[1]


            net.eval() #评估模式
            acc_sum += (net(X.to(device)).argmax(dim=1) == 
            y.to(device).argmax(dim=1)).float().sum().cpu().item()
            net.train() #改回训练模式

            
            '''
            if  isinstance(net,torch.nn.Module):
                net.eval() #评估模式
                acc_sum += (net(X.to(device)).argmax(dim=1) == 
                            y.to(device)).float().sum().cpu().item()
                net.train() #改回训练模式
            else: #自定义模型  3.13节之后不会用到,不考虑GPU
                if( 'is_training ' in net.__code__.co_varnames ):
                    #如果有is_training这个参数
                    #将is_training设置维False
                    acc_sum += (net(X,is_training=False).argmax(dim=1) == y).float.sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
            '''
            #y.shape[0]为一个batch的样本数
            n += y.shape[0]
    return acc_sum/n


import time
def train_CNN(net,train_dataloader,valid_dataloader,batch_size,optimizer,device,num_epochs):
    net = net.to(device)
    print('training on',device)

    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    
   
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum ,n,start = 0.0,0.0,0,time.time()
        all_miss_count = 0
        for i,data in enumerate(train_dataloader):
            print('data shape :',data.shape)
            X = data[0]
            y = data[1]
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu()
            train_acc_sum += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().cpu()
            n += y.shape[0]
            batch_count += 1
            all_miss_count += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().cpu()

        valid_acc = evaluate_accuracy_gpu(net,valid_dataloader,device)
        print('epoch %d , loss %.4f, train acc %.3f, valid acc %.3f, time %.1f sec ' % 
                (epoch+1 , train_l_sum/batch_count ,  train_acc_sum/n , 
                valid_acc,time.time()-start))
