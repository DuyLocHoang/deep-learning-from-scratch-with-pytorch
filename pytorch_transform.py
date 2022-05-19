# # Import
# import torch
# import torchvision
# import torchvision.transforms as transform
# import torchvision.datasets as datasets
# import torch.nn.functional as F
# from torch import nn
# from torch import optim
# from torch.utils.data import DataLoader,Dataset
# import pandas as pd
# from skimage import io
# from torchvision.utils import save_image
# import os
# class CatsAndDogsDataset(Dataset):
#     def __init__(self,csv_file,root_file,transform = None):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_file = root_file
#         self.transform = transform
#     def __len__(self):
#         return len(self.annotations)
#     def __getitem__(self, index):
#         img_path = os.path.join(self.root_file,self.annotations.iloc[index,0])
#         img = io.imread(img_path)
#         y_label = self.annotations.iloc[index,1]
#
#         if self.transform :
#             img = self.transform(img)
#
#         return (img,y_label)
#
# my_transforms = transform.Compose([
#     transform.ToPILImage(),
#     transform.Resize((256,256)),
#     transform.RandomCrop((224,224)),
#     transform.ColorJitter(brightness=0.5),
#     transform.RandomRotation(degrees=45),
#     transform.RandomHorizontalFlip(p=0.5),
#     transform.RandomVerticalFlip(p=0.05),
#     transform.RandomGrayscale(p=0.2),
#     transform.ToTensor(),
#     transform.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])])
#
# dataset = CatsAndDogsDataset(csv_file='dataset/custom/cats_dogs.csv',
#                              root_file="dataset/custom/cats_dogs_resized",
#                              transform=my_transforms)
#
# img_nums = 0
# for _ in range(10):
#     for img,label in dataset :
#         save_image(img,'img'+str(img_nums)+'.png')
#         img_nums += 1
#######################################################################################################################
# Import
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Create model
class CNN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(CNN,self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels=8,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1))
        self.fc1 = nn.Linear(16*8*8,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# Hyperparameter
in_channels = 3
num_classes = 10
learning_rate = 0.005
batch_size = 64
num_epochs = 5
# Load model and modify it
model = CNN(in_channels=in_channels,num_classes = num_classes)
model.classifier = nn.Sequential(nn.Linear(512,100),nn.ReLU(),nn.Linear(100,10))
model.to(device)
print(model)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Load data
my_transforms = transforms.Compose(
    [  # Compose makes it possible to have many transforms
        transforms.Resize((36, 36)),  # Resizes (32,32) to (36,36)
        transforms.RandomCrop((32, 32)),  # Takes a random (32,32) crop
        transforms.ColorJitter(brightness=0.5),  # Change brightness of image
        transforms.RandomRotation(
            degrees=45
        ),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(
            p=0.05
        ),  # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Note: these values aren't optimal
    ]
)
train_dataset = datasets.CIFAR10(root='dataset',train = True, transform=my_transforms,download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
# Train network
for epoch in range(num_epochs):
    losses = []
    for batch_idx,(data,targets) in enumerate(train_loader):
        #Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)
        # Forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        # Backward
        optimizer.zero_grad()
        loss.backward()
        #Gradient descent
        optimizer.step()
    print(f'COst at epoch {epoch} is {sum(losses)/len(losses):.5f}')

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

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader,model)