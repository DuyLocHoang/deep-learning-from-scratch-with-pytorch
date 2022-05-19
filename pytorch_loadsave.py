import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    # Initialize network
    model = torchvision.models.vgg16(pretrained = False)
    optimizer = optim.Adam(model.parameters())

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    # Save checkpoint
    save_checkpoint(checkpoint)

    # Load checkpoint
    load_checkpoint(torch.load("my_checkpoint.pth.tar"),model,optimizer)

if __name__ == "__main__" :
    main()