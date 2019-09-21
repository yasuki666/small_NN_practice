import  torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import transforms

import net
import dataset
using_net = net.net3
learning_rate = 1e-2
num_epoches = 5

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(using_net.parameters(),lr=learning_rate) #随机梯度下降
for epoch in range(num_epoches):
    print("current epoch:%d"%epoch)
    for i,data in enumerate(dataset.MNIST.train_loader):
        images,labels = data
        images = Variable(images.view(images.size(0),-1))
        labels = Variable(labels)

        outputs = using_net(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 ==0:
            print('currect loss = %.5f'%loss.item())

print("finish")

