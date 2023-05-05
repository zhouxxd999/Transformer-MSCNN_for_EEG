import urllib.request
import numpy as np
import pyedflib
from scipy.signal import cheb1ord,cheby1,filtfilt
from scipy import signal
###--------------------------------------------    获取PhysioNet四分类运动想象数据    ----------------------------------------
'''
数据集网址 : https://physionet.org/content/eegmmidb/1.0.0/
'''
def PhysioNetMIConvert(file_name,show_info=False):
    '''
    Parameters
    ————————————————————————————————————————————————————
        file_name :    str
                    edf文件名

    Return
    ————————————————————————————————————————————————————
        masterSet : 2D array (channels x nsamples)
                    edf文件数据  
                    第1个通道是时间戳
                    第2个通道是标签
                    第3-66个通道是电极通道
    '''
    

    reader = pyedflib.EdfReader(file_name)
    annotations = reader.readAnnotations()
    end_time =  annotations[0][-1]+annotations[1][-1]
    intervals = np.append(annotations[0],end_time)   #删除最后半秒的全零数据，将间隔点在末尾添加124.5标志
    
    timeArray = np.array([round(x,5) for x in np.arange(0,end_time,.00625)])
    time_points = int(end_time*160)
    timeArray = timeArray.reshape(time_points,1)   #一共124.5s，采样率160hz，共19920个数据点

    codes = annotations[2]     #codes为事件标志位
    codeArray = []             #codeArray为每一个数据点所代表的事件标志位
    counter = 1     
    for timeVal in timeArray:
        if timeVal == end_time:
            break   
        elif timeVal / intervals[counter] == 1.0:
            counter += 1

        codeArray.append(codes[counter - 1])
    

    invertCodeArray = np.array(codeArray).reshape(time_points,1)
    numSignals = reader.signals_in_file   #数据的通道数（电极数）
    
    signal_labels = reader.getSignalLabels()  #数据通道的标签（电极标签）

    dataset = np.zeros((numSignals, reader.getNSamples()[0]))
    for signal in np.arange(numSignals):
        dataset[signal, :] = reader.readSignal(signal)

    dataset = dataset[:,:time_points].transpose()


    masterSet = np.concatenate((timeArray,invertCodeArray,dataset),axis=1).swapaxes(0,1)


    if show_info:
        print('annotations \n',annotations)  
        print('intervals values \n',intervals)
        print('codeArray value \n',codeArray)
        print('all channels number :',numSignals)
        print('channels labels :',signal_labels)
        print('all file samples :',reader.getNSamples()[0])
        print('masterSet :',masterSet.shape)

    return masterSet


def extractData(raw_data,time_range=[0,4],sample_rate=160):
    '''
        对从PhysioNetMIConvert函数中取出的数进行信号拆解,输出T1和T2类别的数据
    
    Parameters
    ——————————————————————————————————————————————
        raw_data : 2D array (channels x nsamples)
                    PhysioNetMIConvert函数中提取的原始数据
        time_range : list 
                    取出数据段的起始点和终止点  ,单位s
        sample_rate : float
                    数据的采样率

    Returns
    ——————————————————————————————————————————————
        retval : dict
                T1和T2数据   格式为 3D array (ntrials x nchannels x nsamples)

    '''


    start = sample_rate * time_range[0]
    end = sample_rate * time_range[1]
    left_data = []
    idx = 0
    data = raw_data
    while data.shape[1]>0: 
        marker = data[1,:].tolist()
        if 'T1' not in marker:
            break
        idx = marker.index('T1')
        #print('idx : ',idx,'   data shape : ',data.shape)
        l_data = []
        i = idx 
        while marker[i] == 'T1':
            l_data.append(data[2:,i])
            i = i + 1
            if i >= data.shape[1]:
                break
        idx = i
        data = data[:,idx:]
        if len(l_data) >= int(end-start) : 
            left_data.append(l_data[start:end])

    left_data_np = np.array(left_data,dtype=np.float32)
    left_data_np = left_data_np.swapaxes(1,2)


    right_data = []
    idx = 0
    data = raw_data
    while data.shape[1]>0: 
        marker = data[1,:].tolist()
        if 'T2' not in marker:
            break
        idx = marker.index('T2')
        
        r_data = []
        i = idx 
        while marker[i] == 'T2':
            r_data.append(data[2:,i])
            i = i + 1
            if i >= data.shape[1]:
                break
        idx = i
        data = data[:,idx:]
        if len(r_data) >= int(end-start) :
            right_data.append(r_data[start:end])

    right_data_np = np.array(right_data,dtype=np.float32)
    right_data_np = right_data_np.swapaxes(1,2)


    retval = {'T1':left_data_np,'T2':right_data_np}

    return retval

from tqdm import tqdm 
def concatenateAllData(base_dir,num_sub=109):
    '''
        获取全部被试数据  剔除第88,第92和第100个数据异常的被试
    Parameters
    ————————————————————————————————————————————————
    base_dir : str
                基础数据集文件夹
    
    Returns
    ————————————————————————————————————————————————
    all_left : List 
            左手 的全部run的数据 all_left[n]表示第n个被试数据  3D array (ntrials x nchannels x nsamples) 
    all_right : List
    all_fist : List
    all_feet : List
    
    '''
    left_right_runs = [3,4,7,8,11,12]
    fist_feet_runs = [5,6,9,10,13,14]

    all_left = []
    all_right = []
    all_fist = []
    all_feet = []
    for i in tqdm(range(num_sub),ncols=50):
        
        nsub = i + 1
        #print(nsub)
        if nsub == 92 or nsub==100 or nsub==88:

            continue

        for nrun in range(3,15):
            
            sub_file_name = base_dir + '/S' + '{:03d}'.format(nsub) + '/S' +  '{:03d}'.format(nsub) + 'R' + '{:02d}'.format(nrun) + '.edf'
            #print(sub_file_name)
            data = extractData(PhysioNetMIConvert(sub_file_name))
            if nrun in left_right_runs:
                if nrun == 3:
                    left = data['T1']
                    right = data['T2']
                else:
                    left = np.concatenate((left,data['T1']))
                    right = np.concatenate((right,data['T2']))
            elif nrun in fist_feet_runs:
                if nrun == 5:
                    fist = data['T1']
                    feet = data['T2']
                else:
                    fist = np.concatenate((fist,data['T1']))
                    feet = np.concatenate((feet,data['T2']))

        all_left.append(left)
        all_right.append(right)
        all_fist.append(fist)
        all_feet.append(feet) 
        
    print(len(all_left),len(all_right),len(all_fist),len(all_feet))

    return all_left,all_right,all_fist,all_feet                 
                

def getAllSubClassData(left_data,right_data,fist_data,feet_data):
    '''
        获取四个类别所有被试的数据

    Parameters
    ——————————————————————————————————————————
    left_data : List
        左手 的全部run的数据 all_left[n]表示第n个被试数据  3D array (ntrials x nchannels x nsamples) 
    all_right : List
    all_fist : List
    all_feet : List

    Returns 
    ——————————————————————————————————————————
    allSubLeftData : 3D array (all_trials x channels x nsamples)
            所有被试的左手数据
    allSubRightData : 3D array
    allSubFistData : 3D array
    allSubFeetData : 3D array
    '''
    for nsub in range(len(left_data)):
        if nsub == 0:
            allSubLeftData = left_data[nsub]
        else:
            allSubLeftData = np.concatenate((allSubLeftData,left_data[nsub]),axis=0)
    print('allSubLeftData shape : ',allSubLeftData.shape)

    for nsub in range(len(right_data)):
        if nsub == 0:
            allSubRightData = right_data[nsub]
        else:
            allSubRightData = np.concatenate((allSubRightData,right_data[nsub]),axis=0)
    print('allSubRightData shape : ',allSubRightData.shape)

    for nsub in range(len(fist_data)):
        if nsub == 0:
            allSubFistData = fist_data[nsub]
        else:
            allSubFistData = np.concatenate((allSubFistData,fist_data[nsub]),axis=0)
    print('allSubFistData shape : ',allSubFistData.shape)

    for nsub in range(len(feet_data)):
        if nsub == 0:
            allSubFeetData = feet_data[nsub]
        else:
            allSubFeetData = np.concatenate((allSubFeetData,feet_data[nsub]),axis=0)
    print('allFeetLeftData shape : ',allSubFeetData.shape)

    return allSubLeftData,allSubRightData,allSubFistData,allSubFeetData

###--------------------------------------------    滤波 带通滤波与陷波滤波    ----------------------------------------
def cheb_bandpass_filter(data,low,high,fs):
    '''
        使用切比雪夫1型滤波器进行滤波  

    Parameters
    ————————————————————————————————————
    data : 3D array  (trials x channels x samples)
        输入的EEG数据
    low : float
        带通滤波器的低截止频率
    high : float
        带通滤波器的高截止频率
    fs : float
        数据的采样频率

    Returns
    ————————————————————————————————————
    y : 3D array (trials x channels x samples)
        滤波后的数据
    '''
    num_chans,num_samples = data.shape

    fs2 = fs / 2
    Wp = [low/fs2, high/fs2]   #通带
    Ws = [(low-2)/fs2, (high+10)/fs2]  #阻带
    [N, Wn]= cheb1ord(Wp, Ws, 3, 40)
    [B, A] = cheby1(N, 0.5, Wn,'bandpass')

    y = np.zeros(data.shape)

    for nchans in range(num_chans):
        y[nchans,:] = filtfilt(B, A, data[nchans,:])
    return y

def notch_filter(data,fs):
    '''
        50hz陷波滤波器,去除工频噪声
    Parameters 
    ————————————————————————————————
    data : 3D array (trials x channels x samples)
        EEG数据
    fs : float
        数据的采样频率

    Returns
    ————————————————————————————————
    f_data : 3D array (trials x channels x samples)
        陷波滤波后的数据
    '''

    f0 = 50#工频噪声
    q = 35
    b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    
    f_data = np.zeros(data.shape)
    for ntrial in range(data.shape[0]):
        for nchan in range(data.shape[1]):
            f_data[ntrial,nchan,:] = signal.filtfilt(b, a, data[ntrial, nchan,:])

    return f_data


def data_norm(x):
    u = np.mean(x,axis=1,keepdims=True)
    s = np.std(x,axis=1,keepdims=True)
    return (x-u)/s


import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset

# 定义数据读取类
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self,data,label):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
 
        self.data = data
        self.label = label     


    def __getitem__(self, index):
        """
        步骤三:实现__getitem__方法,定义指定index时如何获取数据,并返回单条数据(训练数据，对应的标签)
        """
        eeg_data = torch.tensor(self.data[index],dtype=torch.float32)   
        eeg_label = torch.tensor(self.label[index],dtype=torch.long)

        return eeg_data,eeg_label

    def __len__(self):
        """
        步骤四:实现__len__方法:返回数据集总数目
        """
        return self.label.shape[0]



from models.MSCNN import Config as mscnn_config

class MSCNN_Dataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self,data,label,mscnn_config):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MSCNN_Dataset, self).__init__()
 
        self.data = data
        self.label = label 
        self.fre_start = mscnn_config.freband_start 
        self.fre_range = mscnn_config.freband_range   


    def __getitem__(self, index):
        """
        步骤三:实现__getitem__方法,定义指定index时如何获取数据,并返回单条数据(训练数据，对应的标签)
        """
        eeg_data = self.data[index]
        eeg_data = data_norm(eeg_data)

        r_data = np.zeros((len(self.fre_start),1,eeg_data.shape[0],eeg_data.shape[1]))
        for i in range(len(self.fre_start)):
            r_data[i,0,:,:] = cheb_bandpass_filter(eeg_data,self.fre_start[i],self.fre_start[i]+self.fre_range,fs=160)

        eeg_data = torch.tensor(r_data,dtype=torch.float32)   
        eeg_label = torch.tensor(self.label[index],dtype=torch.long)

        return eeg_data,eeg_label

    def __len__(self):
        """
        步骤四:实现__len__方法:返回数据集总数目
        """
        return self.label.shape[0]





