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


    print("train_data:", train_dataset.data.size())
    print("train_labels:", train_dataset.targets.size())
    print("test_data:", test_dataset.data.size())
    print("test_labels:", test_dataset.targets.size())
