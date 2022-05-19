# Import
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create model
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers= num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_lenght,num_classes)
    def forward(self,x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        # Forward progate lSTM
        out, _ = self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)
        # Decode the hidden states of the last time step
        out = self.fc(out)
        return out
class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(GRU,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_lenght,num_classes)
    def foward(self,x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        # Forward propagate LSTM
        out,_ = self.gru(x,h0)
        out = out.reshape(out.shape[0], -1)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_lenght,num_classes)
    def forward(self,x):
        # Set initial hidden and cell state
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        # Forward propagate LSTM
        out,_ = self.lstm(x,(h0,c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out




# HyperParameter
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
sequence_lenght = 28
learning_rate = 0.005
num_epochs = 5
batch_size = 64
# Load data
train_dataset = datasets.MNIST(root = 'dataset',train = True, transform= transforms.ToTensor(),download=False)
test_dataset = datasets.MNIST(root = 'dataset',train = False, transform= transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle= False)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle= False)
# Initialize model
model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)
# gru = GRU()
# lstm = LSTM()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")