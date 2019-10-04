import torch
import torch.nn as nn
import math

cfg = {'VCG16':[64,64,'M',128,128,'M',256,256,'M',512,512,512,'M',512,512,512,'M']}

class VCG(nn.Module):
    def __init__(self,net_name):
        super(VCG,self).__init__()

        #构建卷积层和池化层
        self.features = self._make_layers(cfg[net_name])

        #构建全连接层和分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
        #初始化权重
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)           #前向传播时先经过卷积层和池化层
        x = x.view(x.size(0),-1)
        x = self.classifier(x)         #将features的结果拼接到分类器上
        return x

    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
                #layers += [conv2d,nn.ReLU(inplace=True)]
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

net = VCG('VCG16')

