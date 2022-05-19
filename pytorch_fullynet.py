
# Import
import torch
import torchvision # torch package for vision
import torch.nn.functional as F  # Parameterless functions, like activations functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # we can peform on our datasets for augmentation
from torch import optim # for optimiziers like adam,SGD,..
from torch import nn # All neurals network module
from torch.utils.data import DataLoader # Give easier dataset managment by creating mini batches
from tqdm import tqdm # For nice process bar

# Create Fully Connected Network
class NN(nn.Module) :
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        # Input 28x28 = > 784
        # First layer : Input size
        # Second layer : Linear (50 notes ) = > output 10 classes
        self.fc1 = nn.Linear(in_features= input_size,out_features= 50)
        self.fc2 = nn.Linear(in_features= 50, out_features= num_classes)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load data
train_dataset = datasets.MNIST(root='dataset/',train=True,transform= transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root = 'dataset/',train= False, transform = transforms.ToTensor(),download = True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle = True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle = True)

# Initialize network
model = NN(input_size= input_size, num_classes = num_classes).to(device)

# Loss and Optimizer
critetion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(tqdm(train_loader)):
        #Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)

        # Get to correct shape
        data = data.reshape(data.shape[0],-1)

        #Forward
        scores = model(data)
        loss =critetion(scores,targets)

        #Backward
        optimizer.zero_grad()
        loss.backward()

        #Gradient descent or adam step
        optimizer.step()

# Ckeck accuracy on training and test to see how good your model
def check_accuracy(loader,model) :
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader :
            #Get data to cuda if possible
            x = x.to(device = device)
            y = y.to(device = device)
            # Get to correct data
            x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _,predictions = scores.max(1)
            num_corrects += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return float(num_corrects)/float(num_samples)

print(f"Accuracy on training set : {check_accuracy(train_loader,model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader,model)*100:.2f}")

# # Import
# # Create model
# # Set device
# # Hyperparameter
# # Load data
# # Initialize model
# # Loss and optimizer
# # Train model
# # Check acc
#
# # Import
# import torch
# import torchvision # torch package for vision
# import torch.nn.functional as F  # Parameterless functions, like activations functions
# import torchvision.datasets as datasets  # Standard datasets
# import torchvision.transforms as transforms  # we can peform on our datasets for augmentation
# from torch import optim # for optimiziers like adam,SGD,..
# from torch import nn # All neurals network module
# from torch.utils.data import DataLoader # Give easier dataset managment by creating mini batches
# from tqdm import tqdm # For nice process bar
# # Import
# import torch
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as tranforms  #
# import torch.nn.functional as F # activation function : relu
# from torch import nn #
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from torch import optim
# class NN(nn.Module) :
#     def __init__(self,input_size,num_classes):
#         super(NN,self).__init__()
#         # Input 28x28 = > 784
#         # First layer : Input size
#         # Second layer : Linear (50 notes ) = > output 10 classes
#         self.fc1 = nn.Linear(in_features= input_size,out_features= 50)
#         self.fc2 = nn.Linear(in_features= 50, out_features= num_classes)
#     def forward(self,x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# # Create model
# class NN(nn.Module):
#     def __init__(self,input_size,num_classes):
#         super(NN,self).__init()
#         self.fc1 = nn.Linear(input_size,50)
#         self.fc2 = nn.Linear(50,num_classes)
#     def forward(self,x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Hyperparameter
# input_size = 784
# num_classes = 10
# learning_rate = 0.001
# num_epochs = 3
# batch_size = 64
# # Load data
# train_dataset = datasets.MNIST(root = 'dataset/',train = True,transform=transforms.ToTensor(),download=True)
# test_dataset = datasets.MNIST(root = 'dataset/',train = False, transform=transforms.ToTensor(),download=True)
# train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle = True)
# test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle = True)
# # Initialize model
# model  = NN(input_size = input_size, num_classes = num_classes).to(device)
#
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(),lr= learning_rate)
# # Train network
# for epoch in range(num_epochs):
#     for batch_idx,(data,targets) in enumerate(train_loader) :
#         # Get data to gpu if possible
#         data = data.to(device = device)
#         targets = targets.to(device = device)
#         # Get to correct data
#         data = data.shape(data.shape[0],-1)
#
#         #Forward
#         scores = model(data)
#         loss = critetion(scores,targets)
#
#         #Backward
#         optimizer.zero_grad()
#         loss.backward()
#
#         #Gradient descent
#         optimizer.step()
# # Check acc
# def check_accuracy(loader,model):
#     num_corrects = 0
#     num_samples = 0
#     model.eval()
#
#     for x,y in loader:
#         x = x.to(device = device)
#         y = y.to(device = device)
#         x = x.reshape(x.shape[0],-1)
#         scores = model(x)
#         _,predictions = scores.max(1)
#         num_corrects += (predictions == y).sum()
#         num_samples += predictions.size(0)
#     model.train()
#     return float(num_corrects)/float(num_samples)
#
# print(f'Accuracy on training set: {check_accuracy(train_loader,model)*100}%')
# print(f"Accracy on test set: {check_accuracy(test_loader,model)*100}%")