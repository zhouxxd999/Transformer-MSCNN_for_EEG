import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from IPython import display
import d2lzh_pytorch as d2l
import random
import torch.nn as nn
import time
import torch.nn.functional as F


#-----------------------------------------------------------------------------------------------------------------------
#                                               线性回归
#-----------------------------------------------------------------------------------------------------------------------
def use_svg_display():
    """使用矢量图表示"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.4,2.5)):
    use_svg_display()
    #设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #样本读取顺序是随机的
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)

def linreg(X,w,b):
    return torch.mm(X,w)+b

def squared_loss(y_hat,y):
    #注意这里返回的是向量，另外pytorch中的MSELoss没有除以2
    return (y_hat-y.view(y_hat.size()))**2 / 2

def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


#-----------------------------------------------------------------------------------------------------------------------
#                                               softmax回归
#-----------------------------------------------------------------------------------------------------------------------

def load_data_fashion_mnist(batch_size,resize=None):
    trans = []
    if resize != None :
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='../data',
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data',
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    train_iter = data.DataLoader(mnist_train,batch_size = batch_size,shuffle=True,num_workers=0)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter,test_iter

def get_fashion_mnist_labels(labels):
    """获取Fashion-MNIST数据集10个类别的标签名称"""
    text_labels = ['t-shirt','trousers','pullover','dress','coat','sandle',
                   'shirt','sneaker','bag','ankle','boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images,labels,row=1):
    d2l.use_svg_display()
    #这里的_表示我们忽略不使用的变量
    image_len = int( len(images) / row )

    _,figs = plt.subplots(row,image_len,figsize=(12,12))
    figs = figs.ravel()
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view(28,28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤。该函数将被逐步改进：它的完整实现将在“图像增⼴”⼀节中描述
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat =net(X)
            l = loss(y_hat,y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net)

        print('epoch %d ,loss %.4f , train acc %.3f , test acc %.3f' % (epoch+1,train_l_sum/n,
                                                                        train_acc_sum/n,test_acc))


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.reshape(x.shape[0],-1)






#-----------------------------------------------------------------------------------------------------------------------
#                                               过拟合与欠拟合
#-----------------------------------------------------------------------------------------------------------------------
def semilogy(x_vals,y_vals,axisx_label,axisy_label,x2_vals=None,y2_vals=None,legend=None,figsize=(5,4)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(axisx_label)
    d2l.plt.ylabel(axisy_label)
    d2l.plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals,y2_vals,linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#                                               第十三课  丢弃法
#-----------------------------------------------------------------------------------------------------------------------
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0

    for X,y in data_iter:
        if isinstance(net,torch.nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
            net.train()
        else:
            if('is_training' in net.__code__.co_varnames):
                acc_sum += (net(X,is_training=False).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


#-----------------------------------------------------------------------------------------------------------------------
#                                               第十九课  丢弃法
#-----------------------------------------------------------------------------------------------------------------------
def corr2d(x,k):
    h,w = k.shape
    Y = torch.zeros((x.shape[0] - h +1,x.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (x[i:i+h,j:j+w] * k).sum()
    return Y


def evaluate_accuracy_gpu(net,data_iter,
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """使用GPU计算模型再数据集上面的精度"""
    acc_sum , n = 0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            if  isinstance(net,torch.nn.Module):
                net.eval() #评估模式
                acc_sum += (net(X.to(device)).argmax(dim=1) == 
                            y.to(device).argmax(dim=1)).float().sum().cpu().item()
                net.train() #改回训练模式
            else: #自定义模型  3.13节之后不会用到，不考虑GPU
                if( 'is_training ' in net.__code__.co_varnames ):
                    #如果有is_training这个参数
                    #将is_training设置维False
                    acc_sum += (net(X,is_training=False).argmax(dim=1) == y.argmax(dim=1)).float.sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1)==y.argmax(dim=1)).float().sum().item()
            #y.shape[0]为一个batch的样本数
            n += y.shape[0]
    return acc_sum/n


def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net = net.to(device)
    print('training on',device)

    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0

    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum ,n,start = 0.0,0.0,0,time.time()
        for X,y in train_iter:
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
        test_acc = evaluate_accuracy_gpu(net,test_iter,device)
        print('epoch %d , loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % 
                (epoch+1 , train_l_sum/batch_count ,  train_acc_sum/n , 
                test_acc,time.time()-start))
    
    
        
            



class GlobalAvgPool2d(nn.Module):
    #全局平均池化可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])



