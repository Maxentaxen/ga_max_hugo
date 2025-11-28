import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.satellite_dataset import SatelliteDataset
from models.convlstm_network import ConvLSTMNetwork

def find_latest_checkpoint(save_dir):
    files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_checkpoint(path, model, optimizer=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in chk:
        optimizer.load_state_dict(chk['optimizer_state_dict'])
    start_epoch = chk.get('epoch', 0)
    return start_epoch, chk.get('loss', None)

def train_model(model, dataloader, criterion, optimizer, num_epochs, save_dir='checkpoints', device=None, start_epoch=0):
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    if device is None:
        device = next(model.parameters()).device
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        # save checkpoint (model + optimizer + epoch + loss)
        chk = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }
        torch.save(chk, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def main():
    
    dataset = SatelliteDataset(r"C:\Users\Max\Documents\GitHub\ga_max_hugo\conv_lstm_satellite\data")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMNetwork(input_dim=1, hidden_dim=64, output_dim=1, kernel_size=3, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    save_dir = 'checkpoints'
    latest = find_latest_checkpoint(save_dir)
    start_epoch = 0
    if latest:
        start_epoch, last_loss = load_checkpoint(latest, model, optimizer, device)
        print(f"Resuming from {latest}, starting at epoch {start_epoch}")

    # num_epochs is total epochs to run (e.g. 50). training will start at start_epoch.
    train_model(model, dataloader, criterion, optimizer, num_epochs=50, save_dir=save_dir, device=device, start_epoch=start_epoch)

if __name__ == '__main__':
    main()
