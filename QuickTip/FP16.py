# Giam 500% Vram giup nang cao hieu suat train


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

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# Hyperparameter
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='dataset/',train = True, transform= transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root = 'dataset/',train = False, transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# Initialize network
model = CNN(in_channels = in_channels, num_classes = num_classes).to(device = device)
# Loss and Optimizier
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        #Forward
        # scores = model(data)
        # loss = criterion(scores,targets)
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, targets)
        #Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Gradient descend
        scaler.step(optimizer)
# Check acc
def check_acc(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    test1 = []
    test2 = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            print(num_correct)
            num_samples += predictions.size(0)
            print(num_samples)
    model.train()
    return float(num_correct)/float(num_samples)

print(f"Accuracy on training set: {check_acc(train_loader,model)*100}")
print(f"Accuracy on training set: {check_acc(test_loader,model)*100}")