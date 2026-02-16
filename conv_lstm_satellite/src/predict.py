import os 
import sys
import glob
import torch 
import argparse 
import time
import threading
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from models.convlstm_network import ConvLSTMNetwork

t_start = time.time()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv('SATELLITE_DATA_DIR', PROJECT_ROOT / 'data'))
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'checkpoint_epoch_50.pth'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

fig, axes = plt.subplots(1, 3, figsize=(12,6))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
stop = threading.Event()


def type_text(text, delay=0.1):
    i = 0
    while not stop.is_set():
        sys.stdout.write(text[i % len(text)])
        sys.stdout.flush()
        time.sleep(delay)
        i += 1

def load_checkpoint(path, model, device=None):
    if device is None:
        device = next(model.parameters()).device
    chk = torch.load(path, map_location=device, weights_only=True)
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



def predict(date_hour, device=None, cmap='viridis', print_ascii=1):
    t_start = time.time()
    folder_path = DATA_DIR / date_hour

    input_seq, target_img = load_data_folder(str(folder_path))
    print(f"Images Loaded")
    
    model = ConvLSTMNetwork(input_dim=1, hidden_dim=64, output_dim=1, kernel_size=3, num_layers=3)
    model = model.to(device)
    model.eval()
    
    epoch = load_checkpoint(CHECKPOINT_PATH, model, device)
    print(f"Model Loaded")

    t = threading.Thread(target=type_text, args=("Cooking...",), daemon=True)
    t.start()

    with torch.no_grad():
        output = model(input_seq.unsqueeze(0).to(device))
        pred = output.squeeze().detach().cpu()
    t_end = time.time()
    stop.set()
    print('\n')

    diff_img = create_difference_image(pred, target_img)
    pred_img = tensor_to_image(pred)
    target_img_pil = tensor_to_image(target_img)

    mae = np.mean(np.abs(pred.numpy() - target_img.numpy()))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    date_array = date_hour.split('/')
    pictures = [target_img_pil, pred_img, diff_img]
    titles = ["Target", "Prediction", "Difference"]
    plt.suptitle(f'Date: {date_array[2]}  {months[int(date_array[1]) - 1]}  {date_array[0]} {date_array[3]}:00 \n MAE: {mae}')

    for pic in range(0,3):
        ax = axes[pic]
        ax.imshow((pictures[pic]), cmap=cmap)
        ax.axis('off')
        ax.set_title(titles[pic])
    return t_end - t_start

if __name__ == '__main__':
    os.system("cls" if os.name == "nt" else "clear")
    parser = argparse.ArgumentParser(description='Run inference on a satellite data folder.')
    parser.add_argument('date_hour', type=str, help='Date and hour in format YYYY/MM/DD/HH (e.g., 2024/09/21/15)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu). If not provided, auto-detects.')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to data directory (e.g., /Volumes/USB_DRIVE/data or D:/USB_DRIVE/data)')
    parser.add_argument('--cmap', type=str, default='gray')
    parser.add_argument('--p', type=int, default=0)
    args = parser.parse_args()
    if args.p:
        print("""  __  __                      _                                    _     
 |  \/  | __ ___  _____ _ __ | |_ __ ___  _____ _ __     ___   ___| |__  
 | |\/| |/ _` \ \/ / _ \ '_ \| __/ _` \ \/ / _ \ '_ \   / _ \ / __| '_ \ 
 | |  | | (_| |>  <  __/ | | | || (_| |>  <  __/ | | | | (_) | (__| | | |
 |_|  |_|\__,_/_/\_\___|_| |_|\__\__,_/_/\_\___|_| |_|  \___/ \___|_| |_|
 | | | |_   _  __ _  __ _  ___ _ __  _ __ ___                            
 | |_| | | | |/ _` |/ _` |/ _ \ '_ \| '__/ _ \                           
 |  _  | |_| | (_| | (_| |  __/ |_) | | | (_) |                          
 |_| |_|\__,_|\__, |\__, |\___| .__/|_|  \___/                           
  _ __  _ __ _|___/_|___/_ _ _|_| |_ ___ _ __ __ _ _ __                  
 | '_ \| '__/ _ \/ __|/ _ \ '_ \| __/ _ \ '__/ _` | '__|                 
 | |_) | | |  __/\__ \  __/ | | | ||  __/ | | (_| | |_ _ _               
 | .__/|_|  \___||___/\___|_| |_|\__\___|_|  \__,_|_(_|_|_)              
 |_|                                                                     """)
        time.sleep(2)
        print("""   ____ _     ___  _   _ ____   ____ _        _    _   _ _  _______ ____    _____  ___   ___   ___  
  / ___| |   / _ \| | | |  _ \ / ___| |      / \  | \ | | |/ / ____|  _ \  |___ / / _ \ / _ \ / _ \ 
 | |   | |  | | | | | | | | | | |   | |     / _ \ |  \| | ' /|  _| | |_) |   |_ \| | | | | | | | | |
 | |___| |__| |_| | |_| | |_| | |___| |___ / ___ \| |\  | . \| |___|  _ <   ___) | |_| | |_| | |_| |
  \____|_____\___/ \___/|____/ \____|_____/_/   \_\_| \_|_|\_\_____|_| \_\ |____/ \___/ \___/ \___/ 
                                                                                                    """)
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    device = torch.device(args.device) if args.device else None
    elapsed_time = predict(args.date_hour, device=device, cmap=args.cmap, print_ascii=args.p)
    print(f'Time elapsed: {elapsed_time} seconds')
    plt.show()
    os.system("cls" if os.name == "nt" else "clear")