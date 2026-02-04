import os, sys, glob, torch, argparse, time
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

t_start = time.time()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv('SATELLITE_DATA_DIR', PROJECT_ROOT / 'data'))
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'checkpoint_epoch_50.pth'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

fig, axes = plt.subplots(1, 3, figsize=(12,6))
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



def predict(date_hour, device=None, cmap='viridis', print_ascii=1):
    folder_path = DATA_DIR / date_hour
    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")
    if print_ascii:
        print(""" _   _                        ____  ____   ___               _       __  __                      _                                                        _                         
    | | | |_   _  __ _  __ _  ___|  _ \|  _ \ / _ \    ___   ___| |__   |  \/  | __ ___  _____ _ __ | |_ __ ___  _____ _ __    _ __  _ __ ___  ___  ___ _ __ | |_ ___ _ __ __ _ _ __    
    | |_| | | | |/ _` |/ _` |/ _ \ |_) | |_) | | | |  / _ \ / __| '_ \  | |\/| |/ _` \ \/ / _ \ '_ \| __/ _` \ \/ / _ \ '_ \  | '_ \| '__/ _ \/ __|/ _ \ '_ \| __/ _ \ '__/ _` | '__|   
    |  _  | |_| | (_| | (_| |  __/  __/|  _ <| |_| | | (_) | (__| | | | | |  | | (_| |>  <  __/ | | | || (_| |>  <  __/ | | | | |_) | | |  __/\__ \  __/ | | | ||  __/ | | (_| | |_ _ _ 
    |_| |_|\__,_|\__, |\__, |\___|_|   |_| \_\\___/   \___/ \___|_| |_| |_|  |_|\__,_/_/\_\___|_| |_|\__\__,_/_/\_\___|_| |_| | .__/|_|  \___||___/\___|_| |_|\__\___|_|  \__,_|_(_|_|_)
                |___/ |___/                                                                                                  |_|                                                       """)
        time.sleep(2)
        print("""   _____ _      ____  _    _ _____   _____ _               _   _ _  ________ _____    ____   ___   ___   ___  
    / ____| |    / __ \| |  | |  __ \ / ____| |        /\   | \ | | |/ /  ____|  __ \  |___ \ / _ \ / _ \ / _ \ 
    | |    | |   | |  | | |  | | |  | | |    | |       /  \  |  \| | ' /| |__  | |__) |   __) | | | | | | | | | |
    | |    | |   | |  | | |  | | |  | | |    | |      / /\ \ | . ` |  < |  __| |  _  /   |__ <| | | | | | | | | |
    | |____| |___| |__| | |__| | |__| | |____| |____ / ____ \| |\  | . \| |____| | \ \   ___) | |_| | |_| | |_| |
    \_____|______\____/ \____/|_____/ \_____|______/_/    \_\_| \_|_|\_\______|_|  \_\ |____/ \___/ \___/ \___/ 
                                                                                                                
                                                                                                                """)
        time.sleep(2)

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
    

    diff_img = create_difference_image(pred, target_img)
    pred_img = tensor_to_image(pred)
    target_img_pil = tensor_to_image(target_img)

    mae = np.mean(np.abs(pred.numpy() - target_img.numpy()))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    date_array = date_hour.split('/')
    pictures = [target_img_pil, pred_img, diff_img]
    titles = ["Target", "Prediction", "Difference"]
    fig.supxlabel(f'Date: {date_array[2]}  {months[int(date_array[1]) - 1]}  {date_array[0]} {date_array[3]}:00 \n Mae: {mae}')

    for pic in range(0,3):
        ax = axes[pic]
        ax.imshow((pictures[pic]), cmap=cmap)
        ax.axis('off')
        ax.set_title(titles[pic])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a satellite data folder.')
    parser.add_argument('date_hour', type=str, help='Date and hour in format YYYY/MM/DD/HH (e.g., 2024/09/21/15)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). If not provided, auto-detects.')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to data directory (e.g., /Volumes/USB_DRIVE/data or D:/USB_DRIVE/data)')
    parser.add_argument('--cmap', type=str, default='viridis')
    parser.add_argument('--p', type=int, default=1)
    args = parser.parse_args()
    
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    device = torch.device(args.device) if args.device else None
    predict(args.date_hour, device=device, cmap=args.cmap, print_ascii=args.p)
    print(f'Time elapsed: {time.time() - t_start - args.p*4} seconds')
    plt.show()
    plt.close('all')