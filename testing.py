import dataset
import  net
import training
import  torch
from torch.autograd import Variable


total = 0
correct =0

for images,labels in dataset.MNIST.test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = training.using_net_1(images)

    _,predicts = torch.max(outputs.data,1)
    total += labels.size(0)
    correct  += (predicts == labels).sum()

print(total)
print(correct)
print("net1 Accuracy: %.2f " % (100*correct/total)+"%")


total = 0
correct =0

for images,labels in dataset.MNIST.test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = training.using_net_2(images)

    _,predicts = torch.max(outputs.data,1)
    total += labels.size(0)
    correct  += (predicts == labels).sum()

print(total)
print(correct)
print("net2 Accuracy: %.2f " % (100*correct/total)+"%")


total = 0
correct =0

for images,labels in dataset.MNIST.test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = training.using_net_3(images)

    _,predicts = torch.max(outputs.data,1)
    total += labels.size(0)
    correct  += (predicts == labels).sum()

print(total)
print(correct)
print("net3 Accuracy: %.2f " % (100*correct/total)+"%")