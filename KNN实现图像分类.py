
#未归一化，正确率不高
#基于MNIST数据集

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import KNN_classify_algorithm as KNN
batch_size = 100

#MNIST dataset
train_dataset = dsets.MNIST(root='/dataset/MNIST',train = True,transform = None,download=True)

test_dataset = dsets.MNIST(root='/dataset/MNIST',train = False,transform = None,download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size = batch_size,shuffle=True)
'''
print("train_data:",train_dataset.data.size())
print("train_labels:",train_dataset.targets.size())
print("test_data:",test_dataset.data.size())
print("test_labels:",test_dataset.targets.size())
'''

'''
digit = train_loader.dataset.train_data[0]
plt.imshow(digit,cmap = plt.cm.binary)
plt.show()
print(train_loader.dataset.train_labels[0])
'''

X_train = train_loader.dataset.data.numpy()
X_train = X_train.reshape(X_train.shape[0],28*28)
Y_train = train_loader.dataset.targets.numpy()
X_test = test_loader.dataset.data[:1000].numpy()
X_test = X_test.reshape(X_test.shape[0],28*28)
Y_test = test_loader.dataset.targets[:1000].numpy()
num_test = Y_test.shape[0]
Y_test_pred = KNN.KNN_classify(5,'M',X_train,Y_train,X_test)
num_correct = np.sum(Y_test == Y_test_pred)
accuracy = float(num_correct)/num_test

print("GOT {} /{} correct => accuracy: {}".format(num_correct,num_test,accuracy))