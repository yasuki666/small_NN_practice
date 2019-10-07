import torch
import  torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import small_CNN_practice.net1 as net1
net = net1.net
checkpoint = torch.load(r'C:\Users\11038\PycharmProjects\small NN practice\models\Cifar10_model\6.ckpt')
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

def main():
    batch_size = 4
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root='/dataset/Cifar10', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='/dataset/Cifar10', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

    for epoch in range(7):
        train_loss = 0.0
        for batch_index, data in enumerate(train_loader,0):

            inputs,labels = data
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            #查看训练状态
            train_loss += loss.item()
            if batch_index % 2000 == 1999:
                print('[%d,%5d] loss:%.3f' % (epoch+1,batch_index+1,train_loss/2000))
                train_loss = 0.0
        print('Saving epoch %d model...' % (epoch+1))
        state = {
                'net':net.state_dict(),
                'epoch':epoch+1
            }

        torch.save(state,r'C:\Users\11038\PycharmProjects\small NN practice\models\Cifar10_model\%d.ckpt'%(epoch+7))
        print('saved %d epoch'%(epoch+1))
    print('finish')

if __name__ == '__main__':
    main()