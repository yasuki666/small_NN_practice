import  torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import transforms

import net
import dataset
using_net_1 = net.net1
using_net_2 = net.net2
using_net_3 = net.net3
learning_rate = 1e-2
num_epoches = 5




#训练简单神经网络
print("开始训练net1")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(using_net_1.parameters(),lr=learning_rate) #随机梯度下降
for epoch in range(num_epoches):
    print("current epoch:%d"%epoch)
    for i,data in enumerate(dataset.MNIST.train_loader):
        images,labels = data
        images = Variable(images.view(images.size(0),-1))
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = using_net_1(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 ==0:
            print('currect loss = %.5f'%loss.item())


print("开始训练net2")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(using_net_2.parameters(),lr=learning_rate) #随机梯度下降
for epoch in range(num_epoches):
    print("current epoch:%d"%epoch)
    for i,data in enumerate(dataset.MNIST.train_loader):
        images,labels = data
        images = Variable(images.view(images.size(0),-1))
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = using_net_2(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 ==0:
            print('currect loss = %.5f'%loss.item())

print("开始训练net3")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(using_net_3.parameters(),lr=learning_rate) #随机梯度下降
for epoch in range(num_epoches):
    print("current epoch:%d"%epoch)
    for i,data in enumerate(dataset.MNIST.train_loader):
        images,labels = data
        images = Variable(images.view(images.size(0),-1))
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = using_net_3(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 ==0:
            print('currect loss = %.5f'%loss.item())

    state  = {
        'net':using_net_3.state_dict(),
        'epoch': epoch+1,
    }
    torch.save(state,r'C:\Users\11038\PycharmProjects\small NN practice\models\%d' % (epoch+1))

print("finish")

