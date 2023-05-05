import numpy as np
from scipy import signal 
from scipy.signal import cheb1ord,cheby1,filtfilt

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
    num_trials, num_chans,num_samples = data.shape

    fs2 = fs / 2
    Wp = [low/fs2, high/fs2]   #通带
    Ws = [(low-2)/fs2, (high+10)/fs2]  #阻带
    [N, Wn]= cheb1ord(Wp, Ws, 3, 40)
    [B, A] = cheby1(N, 0.5, Wn,'bandpass')

    y = np.zeros(data.shape)

    for ntrials in range(num_trials):
        for nchans in range(num_chans):
            y[ntrials,nchans,:] = filtfilt(B, A, data[ntrials, nchans,:])

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

def preprocess_mean(data):
    '''
        将所有的通道的数据进行全局平均(不同trial的数据全部加进来平均)
    
    Parameters
    ——————————————————————————
    data : 3D array  (trials x nchannels x samples)
            EEG data

    Return 
    ——————————————————————————
    data_centered : 3D array  (trials x nchannels x samples)
            Centralized data
    '''
    ntrials , nchans , nsamples = data.shape
    
    data_centered = np.zeros((ntrials,nchans,nsamples))
    for nc in range(nchans):
        m = np.zeros((ntrials,),dtype=np.float32)
        for nt in range(ntrials):
            m[nt] = np.mean(data[nt,nc,:])
        m_all = np.mean(m)
        data_centered[:,nc,:] = data[:,nc,:] - m_all

    return data_centered

def cal_R_avg(data_cls):
    '''
        计算归一化后的平均协方差矩阵
    
    '''
    Ra = np.zeros((data_cls.shape[0],data_cls.shape[1],data_cls.shape[1]))
    Ra_avg = np.zeros((data_cls.shape[1],data_cls.shape[1]))
    Ra_trace = np.zeros((data_cls.shape[0],))
    for i in range(data_cls.shape[0]):
        Ra[i,:,:] =  np.dot(data_cls[i,:,:],data_cls[i,:,:].T) 
        Ra_avg = Ra_avg + Ra[i,:,:]
        Ra_trace[i] = np.trace(Ra[i,:,:])
    Ra_alltrace = np.sum(Ra_trace)
    Ra_avg = Ra_avg / Ra_alltrace

    return Ra_avg

def cal_csp(data_cls1,data_cls2,num):
    '''
        对两类数据进行CSP计算,返回投影矩阵

    Parameters
    ——————————————————————————
    data_cls1 : 3D array (trials x nchannels x samples)
            data of class1
    data_cls2 : 3D array (trials x nchannels x samples)
            data of class2
    num : int 
            selected feature num ( get feature number is 2*num) 

    Retrun 
    —————————————————————————
    f : 2D array (2*num x nchannels)
        project matrix
    '''

    #将数据中心化
    data_c1 = preprocess_mean(data_cls1)
    data_c2 = preprocess_mean(data_cls2)

    #计算平均协方差矩阵
    Ra = cal_R_avg(data_c1)
    Rb = cal_R_avg(data_c2)
    #print(Ra)
    #print(Rb)
    R = Ra + Rb

    from numpy import linalg as LA
    #print(np.isnan(R).any())
    #print(R)

    w1,v1 = LA.eig(R)


    w1_mean = np.nanmean(w1)
    for i in range(w1.shape[0]):
        if w1[i] < 1.0e-18:
            #print('w[',i,'] too small and rewrite to w1 mean')
            w1[i] = w1_mean
    
    P = np.dot(np.diag(np.power(w1, -0.5)),v1.T)


    Sa = np.dot(P,np.dot(Ra,P.T))
    Sb = np.dot(P,np.dot(Rb,P.T))

    # w, v = LA.eig(LA.pinv(Y1) * Y2)  #w特征值，v特征向量
    w, v = LA.eig(np.dot(LA.pinv(Sa),Sb))


    #print('v value ',v)

    #print('eig value ',w)
    sorted_indices = np.argsort(w)  # 将特征值按从小到大排序
    W = np.mat(v.real).T * P
    #print('W shape :',W.shape)
    f = np.vstack((W[sorted_indices[:-num - 1:-1], :],
                   W[sorted_indices[0:num:][::-1], :]))
    #print('f shape :',f.shape)
    #print('f value :',f)

    return f


def cal_feature(W,data,num):
    '''
        根据投影矩阵,将数据映射,之后获取log(var(projected data))作为特征
    
    Parameters
    ————————————————————————————————————————
    W : 2D array (2*num x nchannels)
        project matrix
    data : 3D array ( ntrials x nchannels x xsamples)
        EEG data
    num : int
        selected feature num ( get feature number is 2*num) 

    Return 
    ————————————————————————————————————————
    v : 1D array
        feature vector
    '''
    
    # data - (trialNum, channelNum,timepoint)
    v = np.zeros((data.shape[0],2*num))
    #print('V. shape is :',v.shape)
    #print('data input shape :',data.shape)
    for i in range(data.shape[0]):
        z = np.dot(W , data[i,:,:])
        #('z shape : ',z.shape)
        #v[i,:] =  np.array(np.log(np.diag(np.dot(z,z.T)) / np.trace(np.dot(z,z.T)))).squeeze()
        v[i,:] =  np.array(np.log(np.diag(np.dot(z,z.T)) / np.trace(np.dot(z,z.T)))).squeeze()
    return v


