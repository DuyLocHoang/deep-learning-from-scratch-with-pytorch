import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST(root='dataset/',train = True, transform= transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
def get_mean_std(loader):
    #VAR[X] = E[X**2] - E[X]**2
    for data,_ in train_loader :
        channels_sum,channel_squared_sum,num_batches = 0,0,0

        channels_sum += torch.mean(data,dim=[0,2,3])
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])
        num_batches +=1

    mean = channels_sum/num_batches
    std = channel_squared_sum/num_batches - mean**2

    return mean,std

mean,std = get_mean_std(train_loader)
print(mean)
print(std)

# x = torch.randn(1,2,3,4)
# print(x)
# print(x[0,0,1,1])
#
# print(torch.mean(x, dim=[0, 2, 3]))
# print(torch.mean(x, [0, 2, 3]))
# tensor([[[[ 0.5062,  0.3141, -0.0122,  0.7622],
#           [ 0.4541,  0.0893,  1.3551,  1.0492],
#           [-0.7966, -1.4413,  0.0609, -0.3250]],
#
#          [[-0.9145,  0.7881,  1.0062,  1.7014],
#           [ 0.0586,  0.8218, -1.2955,  0.7498],
#           [ 0.2287, -0.1889, -1.7497,  1.0748]]]])
# tensor(0.0893)
# tensor([0.1680, 0.1901])
# tensor([0.1680, 0.1901])