import torch
import small_CNN_practice.net1
import small_CNN_practice.train1
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

def main():
    batch_size = 4
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.CIFAR10(root='/dataset/Cifar10', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = small_CNN_practice.net1.net
    checkpoint = torch.load(r'C:\Users\11038\PycharmProjects\small NN practice\models\Cifar10_model\5.ckpt')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

    correct = 0
    total = 0
    with torch.no_grad():
        for data1 in test_loader:
            images1,labels1 = data1
            outputs = net(images1)
            _,predicted = torch.max(outputs.data,1)
            total += labels1.size(0)
            correct += (predicted==labels1).sum().item()

    print('Accuracy of the network on the 10000 test images:%d %%' %(100*correct/total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (cifar10_classes[i],100*class_correct[i]/class_total[i]))

if __name__ == '__main__':
    main()