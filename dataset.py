import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

class MNIST():
    batch_size = 64
    train_dataset = datasets.MNIST(root='/dataset/MNIST',train = True,transform = transforms.ToTensor(),download=True)
    test_dataset = datasets.MNIST(root='/dataset/MNIST',train = False,transform = transforms.ToTensor(),download=True)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Cifar10():
    batch_size = 4
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_dataset = datasets.CIFAR10(root='/dataset/Cifar10',train=True,transform=transform,download=True)
    test_dataset = datasets.CIFAR10(root='/dataset/Cifar10',train=False,transform=transform,download=True)
    train_loader = DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle=True,num_workers=2)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle = False,num_workers=2)
    cifar10_classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

