import torch
from torch import nn

VGG_types = {
    "VGG16": [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

}


class VGG_net(nn.Module):
    def __init__(self,in_channels = 3, num_classes = 1000):
        super(VGG_net,self).__init__()
        self.in_channel = in_channels
        self.num_classes = num_classes
        self.conv_layer = self.create_conv_layers(VGG_types["VGG16"])
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    def forward(self,x):
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        return x
    def create_conv_layers(self,architecture):
        layers = []
        in_channel = self.in_channel

        for x in architecture :
            if type(x) == int :
                out_channel = x
                layers += [nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=(3,3),
                                    padding=(1,1),
                                    stride=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channel = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
        return nn.Sequential(*layers)

if __name__ == "__main__" :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG_net().to(device)
    x = torch.rand(3,3,224,224).to(device)

    print(model(x).shape)


