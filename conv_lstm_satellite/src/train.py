import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.satellite_dataset import SatelliteDataset
from models.convlstm_network import ConvLSTMNetwork
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

def main():
    config = load_config('config.yaml')
    
    dataset = SatelliteDataset(config['dataset']['path'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    model = ConvLSTMNetwork(input_dim=config['model']['input_dim'], hidden_dim=config['model']['hidden_dim'], output_dim=config['model']['output_dim'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    train_model(model, dataloader, criterion, optimizer, config['training']['num_epochs'])

if __name__ == '__main__':
    main()