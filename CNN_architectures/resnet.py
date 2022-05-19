import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,in_channels,intermediate_channels,identity_downsample = None, stride = 1):
        super(Block, self).__init__()
        self.expansions = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels,kernel_size=1,stride = 1,padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels,kernel_size=3,stride = stride,bias=False,padding =1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansions,kernel_size=1,stride = 1,padding =0,bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels*self.expansions)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self,x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.identity_downsample is not None :
            identity =self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self,Block,layers,image_channels,num_classes = 1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64,kernel_size=7,stride = 2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # Tao cac layer cua Resnet-50
        self.layer1 = self._make_layers(Block,3,64,1)
        self.layer2 = self._make_layers(Block,4,128,2)
        self.layer3 = self._make_layers(Block,6,256,2)
        self.layer4 = self._make_layers(Block,3,512,2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048,num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layers(self,Block,num_residual_blocks,intermediate_channels,stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels *4 :
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,intermediate_channels*4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(intermediate_channels*4))
            layers.append(Block(self.in_channels,intermediate_channels,identity_downsample,stride))
            self.in_channels = intermediate_channels*4

            for i in range(num_residual_blocks-1):
                layers.append(Block(self.in_channels,intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels =3,num_classes =1000):
    return ResNet(Block,50,img_channels,num_classes)

def test():
    net = ResNet50(img_channels=3, num_classes=1000)
    print(net)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())

test()