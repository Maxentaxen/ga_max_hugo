import os, sys, glob, torch, argparse, time
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

t_start = time.time()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv('SATELLITE_DATA_DIR', PROJECT_ROOT / 'data'))
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'checkpoint_epoch_50.pth'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from models.convlstm_network import ConvLSTMNetwork

def load_checkpoint(path, model, device=None):
    if device is None:
        device = next(model.parameters()).device
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    return chk.get('epoch', 0)

def load_data_folder(folder_path):
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    
    input_seq = []
    for img_path in image_files[:3]:
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0
        input_seq.append(torch.from_numpy(img).unsqueeze(0))
    input_seq = torch.stack(input_seq, dim=0)
    
    target_img = Image.open(image_files[3]).convert('L')
    target_img = np.array(target_img, dtype=np.float32) / 255.0
    
    return input_seq, torch.from_numpy(target_img)

def tensor_to_image(tensor):
    t = tensor.detach().cpu().squeeze().numpy()
    t = np.clip(t, 0.0, 1.0)
    return Image.fromarray((t * 255.0).astype(np.uint8), mode='L')

def create_difference_image(pred_tensor, target_tensor):
    diff = np.abs(pred_tensor.detach().cpu().squeeze().numpy() - target_tensor.detach().cpu().squeeze().numpy())
    return Image.fromarray((diff * 255.0).astype(np.uint8), mode='L')

def predict(date_hour, device=None):
    folder_path = DATA_DIR / date_hour
    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")
    
    print(f"Loading data...")
    input_seq, target_img = load_data_folder(str(folder_path))
    
    model = ConvLSTMNetwork(input_dim=1, hidden_dim=64, output_dim=1, kernel_size=3, num_layers=3)
    model = model.to(device)
    model.eval()
    
    print(f"Loading model...")
    epoch = load_checkpoint(CHECKPOINT_PATH, model, device)
    
    with torch.no_grad():
        output = model(input_seq.unsqueeze(0).to(device))
        pred = output.squeeze().detach().cpu()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    date_hour_clean = date_hour.replace('/', '-').replace(':', '-')
    paths = {
        'pred': OUTPUT_DIR / 'preds' / f'pred_{date_hour_clean}.png',
        'target': OUTPUT_DIR / 'targets' / f'target_{date_hour_clean}.png',
        'diff': OUTPUT_DIR / 'diffs' / f'diff_{date_hour_clean}.png',
    }
    
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    tensor_to_image(pred).save(str(paths['pred']))
    tensor_to_image(target_img).save(str(paths['target']))
    create_difference_image(pred, target_img).save(str(paths['diff']))
    
    mae = np.mean(np.abs(pred.numpy() - target_img.numpy()))
    rmse = np.sqrt(np.mean((pred.numpy() - target_img.numpy()) ** 2))
    
    print(f"Images saved to: {OUTPUT_DIR}")
    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a satellite data folder.')
    parser.add_argument('date_hour', type=str, help='Date and hour in format YYYY/MM/DD/HH (e.g., 2024/09/21/15)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). If not provided, auto-detects.')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to data directory (e.g., /Volumes/USB_DRIVE/data or D:/USB_DRIVE/data)')
    
    args = parser.parse_args()
    
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    
    device = torch.device(args.device) if args.device else None
    predict(args.date_hour, device=device)
    print(f'Time elapsed: {time.time() - t_start} seconds')