# Import
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import pandas as pd
from skimage import io
import os

# Dataset
class CatsAndDogsDataset(Dataset) :
    def __init__(self,csv_file,root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index,1])
        if self.transform :
            image = self.transform(image)
        return (image,y_label)



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = torchvision.models.googlenet(pretrained=True        )
model.to(device)
# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 5
# Load data
dataset = CatsAndDogsDataset(
    csv_file='dataset/custom/cats_dogs.csv',
    root_dir = 'dataset/custom/cats_dogs_resized',
    transform = transforms.ToTensor()
)
# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.
train_set,test_set = torch.utils.data.random_split(dataset,[5,5])
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle = True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle = True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
# Train network
for epoch in range(num_epochs):
    losses = []
    for batch_idx,(data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        #Forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        #Backward
        optimizer.zero_grad()
        loss.backward()
        #Gradient Descend
        optimizer.step()
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")
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
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} ")

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

