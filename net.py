#创建一个简单的三层全连接神经网络
import torch
from torch import  nn


#简单三层神经网络
class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#加入激活函数的NN
class Activation_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Activation_Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#添加一个快速收敛方法
class Batch_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

net1 = simpleNet(28*28,300,100,10)
net2 = Activation_Net(28*28,300,100,10)
net3 = Batch_Net(28*28,300,100,10)

print(net1)
