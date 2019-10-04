
import dataset
import  net
import  torch
from torch.autograd import Variable


using_net = net.net3
checkpoint = torch.load(r'C:\Users\11038\PycharmProjects\small NN practice\models\MNIST_model\5.ckpt')
using_net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

total = 0
correct =0

for images,labels in dataset.MNIST.test_loader:
    images = Variable(images.view(-1,28*28))

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    outputs = using_net(images)
    _,predicts = torch.max(outputs.data,1)
    total += labels.size(0)
    correct  += (predicts == labels).sum()

print(total)
print(correct)
print("net3 Accuracy: %.2f " % (100*correct/total)+"%")


