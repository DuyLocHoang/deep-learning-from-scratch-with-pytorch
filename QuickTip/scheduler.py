# Import
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameter
num_classes = 10
batch_size = 64
num_epochs = 5
learning_rate = 0.0001
# Model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# Load model and modify it
model = torchvision.models.vgg16(pretrained=False)
# if you want to do finetuning then set required_grad = False
# Remove two line if you want train entire model,and only want to load the pretrain weights
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,num_classes))
model.to(device)
print(model)

# Load data
train_dataset = datasets.CIFAR10(root = 'dataset', transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size= batch_size,shuffle= True)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=5,verbose=True)
# Train network
for epoch in range(num_epochs):
    losses = []
    for batch_idx,(data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        #forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        #Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient descend
        optimizer.step()
    #mean loss
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses):.5f}")
# Check acc
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader :
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}\{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader,model)