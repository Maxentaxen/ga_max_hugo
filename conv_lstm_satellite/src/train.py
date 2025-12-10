import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.satellite_dataset import SatelliteDataset
from models.convlstm_network import ConvLSTMNetwork
from PIL import Image
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
PRED_DIR = PROJECT_ROOT / 'predictions'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
DATA_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

def train_model(model, dataloader, criterion, optimizer, num_epochs, save_dir=str(CHECKPOINT_DIR), device=None, start_epoch=0, dataset=None, pred_dir=str(PRED_DIR)):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    model.train()
    if device is None:
        device = next(model.parameters()).device
    n_batches = len(dataloader)
    if n_batches == 0:
        raise RuntimeError(f"DataLoader has zero batches. Check dataset at: {DATA_DIR}")
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
        epoch_loss = running_loss / n_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        chk = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }
        torch.save(chk, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        latest_path = os.path.join(save_dir, 'latest.pth')
        torch.save({'epoch': epoch + 1, 'path': os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')}, latest_path)
        if dataset is not None:
            gen_image(model, dataset, epoch, device, sample_idx=0, save_dir=pred_dir)

def generate_prediction(model, sample, device):
    prev_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(sample, (tuple, list)):
                inputs = sample[0]
            else:
                inputs = sample
            if isinstance(inputs, torch.Tensor):
                if inputs.ndim == 4:
                    inputs = inputs.unsqueeze(0)
                elif inputs.ndim == 3:
                    inputs = inputs.unsqueeze(0).unsqueeze(1)
            else:
                inputs = torch.as_tensor(inputs, dtype=torch.float32)
                if inputs.ndim == 4:
                    inputs = inputs.unsqueeze(0)
                elif inputs.ndim == 3:
                    inputs = inputs.unsqueeze(0).unsqueeze(1)
            inputs = inputs.to(device)
            output = model(inputs)
    finally:
        model.train(prev_mode)
    return output.detach().cpu()

def process_output(tensor):
    t = tensor.detach().cpu()
    if t.ndim == 5:
        t = t[0, -1]
    elif t.ndim == 4:
        t = t[0]
    if isinstance(t, torch.Tensor):
        arr = t.numpy()
    else:
        arr = np.asarray(t)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 1:
        img = Image.fromarray(arr[0], mode='L')
    elif arr.ndim == 2:
        img = Image.fromarray(arr, mode='L')
    else:
        img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    return img

def save_pred(img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def gen_image(model, dataset, epoch, device, sample_idx=0, save_dir=str(PRED_DIR)):
    try:
        sample = dataset[sample_idx]
    except Exception:
        return
    pred = generate_prediction(model, sample, device)
    img = process_output(pred)
    fname = f'epoch_{epoch+1:03d}_sample_{sample_idx:03d}.png'
    path = os.path.join(save_dir, fname)
    save_pred(img, path)

def main():
    dataset = SatelliteDataset(root_dir=str(DATA_DIR))
    print("Dataset size:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMNetwork(input_dim=1, hidden_dim=64, output_dim=1, kernel_size=3, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    save_dir = str(CHECKPOINT_DIR)
    latest = find_latest_checkpoint(save_dir)
    start_epoch = 0
    if latest:
        start_epoch, last_loss = load_checkpoint(latest, model, optimizer, device)
        print(f"Resuming from {latest}, starting at epoch {start_epoch}")
    train_model(model, dataloader, criterion, optimizer, num_epochs=50, save_dir=save_dir, device=device, start_epoch=start_epoch, dataset=dataset, pred_dir=str(PRED_DIR))

if __name__ == '__main__':
    main()
