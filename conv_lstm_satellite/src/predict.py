import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import time
t_start = time.time()
elapsed_times = []
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Check for data directory from environment variable or use default
DATA_DIR = Path(os.getenv('SATELLITE_DATA_DIR', PROJECT_ROOT / 'data'))
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

OUTPUT_DIR = PROJECT_ROOT / 'predictions'
DIFF_DIR = OUTPUT_DIR / 'diffs'
PREDICTION_DIR = OUTPUT_DIR / 'preds'
TARGET_DIR = OUTPUT_DIR / 'targets'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from dataset.satellite_dataset import SatelliteDataset
from models.convlstm_network import ConvLSTMNetwork

def find_latest_checkpoint(save_dir):
    files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_checkpoint(path, model, device=None):
    if device is None:
        device = next(model.parameters()).device
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    return chk.get('epoch', 0)

def load_data_folder(folder_path):
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if len(image_files) < 4:
        raise FileNotFoundError(f"Expected at least 4 images in {folder_path}, found {len(image_files)}")
    
    input_seq = []
    for img_path in image_files[:3]:
        img = Image.open(img_path).convert('L')
        img = img.resize((256, 256), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        input_seq.append(img)
    input_seq = torch.stack(input_seq, dim=0)
    
    target_img = Image.open(image_files[3]).convert('L')
    target_img = target_img.resize((256, 256), Image.BILINEAR)
    target_img = np.array(target_img, dtype=np.float32) / 255.0
    target_img = torch.from_numpy(target_img)
    
    return input_seq, target_img

def tensor_to_image(tensor):
    t = tensor.detach().cpu()
    if t.ndim > 2:
        t = t.squeeze()
    arr = t.numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode='L')

def create_difference_image(pred_tensor, target_tensor):
    pred = pred_tensor.detach().cpu().numpy()
    target = target_tensor.detach().cpu().numpy()
    
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()
    
    diff = np.abs(pred - target)
    diff = (diff * 255.0).astype(np.uint8)
    return Image.fromarray(diff, mode='L')

def predict(date_hour, checkpoint_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    folder_path = DATA_DIR / date_hour
    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")
    
    print(f"Loading data from: {folder_path}")
    input_seq, target_img = load_data_folder(str(folder_path))
    
    model = ConvLSTMNetwork(input_dim=1, hidden_dim=64, output_dim=1, kernel_size=3, num_layers=3)
    model = model.to(device)
    model.eval()
    
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(str(CHECKPOINT_DIR))

    epoch = load_checkpoint(checkpoint_path, model, device)
    print(f"Loaded model from epoch {epoch}: {checkpoint_path}")
    
    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0).to(device)
        output = model(input_seq)
        pred = output.squeeze().detach().cpu()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pred_img = tensor_to_image(pred)
    target_img_pil = tensor_to_image(target_img)
    diff_img = create_difference_image(pred, target_img)
    
    date_hour_clean = date_hour.replace('/', '-').replace(':', '-')
    pred_path = PREDICTION_DIR / f'pred_{date_hour_clean}.png'
    target_path = TARGET_DIR / f'target_{date_hour_clean}.png'
    diff_path = DIFF_DIR / f'diff_{date_hour_clean}.png'
    
    pred_img.save(str(pred_path))
    target_img_pil.save(str(target_path))
    diff_img.save(str(diff_path))
    
    mae = np.mean(np.abs(pred.numpy() - target_img.numpy()))
    mse = np.mean((pred.numpy() - target_img.numpy()) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"Prediction saved to: {pred_path}")
    print(f"Target saved to: {target_path}")
    print(f"Difference saved to: {diff_path}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    return pred_img, target_img_pil, diff_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a satellite data folder.')
    parser.add_argument('date_hour', type=str, help='Date and hour in format YYYY/MM/DD/HH (e.g., 2024/09/21/15)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file. If not provided, uses latest.')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). If not provided, auto-detects.')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to data directory (e.g., /Volumes/USB_DRIVE/data or D:/USB_DRIVE/data)')
    
    args = parser.parse_args()
    
    # Override DATA_DIR if provided
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    
    device = None
    if args.device:
        device = torch.device(args.device)
    predict(args.date_hour, checkpoint_path=args.checkpoint, device=device)
    t_end = time.time()
    print(f'Time elapsed: {t_end-t_start} seconds')