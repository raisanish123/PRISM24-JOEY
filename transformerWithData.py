import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Set random seeds for reproducibility
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load multiple files together
csv_list = ['combined_H1_edited.csv']
df_occupancy = pd.DataFrame()  # main data storage 
for file in csv_list:
    df_temp = pd.read_csv(file)
    df_occupancy = pd.concat([df_occupancy, df_temp], ignore_index=True)

# Load and preprocess Data
print("File name", csv_list)
print("Occupancy data before dropping NA data", df_occupancy.shape)
df_occupancy = df_occupancy.dropna(how='any', axis=0)
print("Occupancy data after dropping NA data", df_occupancy.shape)
X = df_occupancy.drop(['date', 'occupied', 'number'], axis=1)
print("X shape", X.shape)
# Update y as necessary for occupied column or numbers column
y = df_occupancy['number']

# Standard scaling
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
print("X shape", X.shape)
print("y shape", y.shape)

# Create testing and training sets for occupied
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)

# Convert data to PyTorch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32)).to(device)
        self.y = torch.from_numpy(y.to_numpy().astype(np.int64)).to(device)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

# Instantiate training and test data
train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)

train_loader = DataLoader(dataset=train_data, batch_size=35, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=35, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1).to(device)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, max_seq_length, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length).to(device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers).to(device)
        self.fc_out = nn.Linear(d_model, num_classes).to(device)

    def forward(self, src):
        src = self.embedding(src).to(device)
        src = self.positional_encoding(src).to(device)
        output = self.transformer_encoder(src).to(device)
        output = output.mean(dim=1).to(device)
        output = self.fc_out(output).to(device)
        return output

input_dim = X_train.shape[1]
d_model = 64
nhead = 4
num_layers = 3
dim_feedforward = 256
max_seq_length = input_dim
num_classes = len(y.unique())
dropout = 0.1

model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, max_seq_length, num_classes, dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")

# Validation loop
model.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

val_loss /= len(test_loader)
accuracy = correct / total

print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {accuracy}")

