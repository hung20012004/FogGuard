import cv2
import torch
import math
import glob
import urllib.request
import os
from util.parse_config import parse_data_config
import argparse
import numpy as np

# import util.transforms as utransforms
# from torchvision import utils
# import util.datasets as ds
from tqdm import tqdm
# import numpy as np
# from PIL import Image
# from os.path import expanduser
from pathlib import Path
# home = expanduser("~")
home = str(Path.home())

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_config",
                    default="../config/voc-5.data",
                    help=".data file containing the config of dataset")

parser.add_argument("--data_type",
                    default="train", choices=["train", "test"])

parser.add_argument("--directory",
                    default="/home/soheil/data/VOC/train/VOCdevkit/VOC2012/images")

flags = parser.parse_args()

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


batch_size = 32
image_size = 416

data_config = parse_data_config(flags.data_config)
# simple_transform = utransforms.simple_transform(image_size)

# train_ds = ds.yolo_dataset(data_config, "train", simple_transform, image_size=image_size,
#                             batch_size=batch_size,
#                             # num_samples,
#                             )
# train_dl, valid_dl = train_ds.create_dataloader()

# for _, imgs, targets in train_dl:
#     imgs = transform(imgs).to(device)
#     with torch.no_grad():
#         prediction = midas(imgs)

# input_batch = transform(img).to(device)
# images_list_dir = data_config['test']
# images_list_dir = data_config['train']
images_list_dir = home + data_config[flags.data_type]
print("Image list file", images_list_dir)

# Fix path handling for Windows
print(f"Original path: {repr(images_list_dir)}")

if os.name == 'nt':  # Windows
    images_list_dir = images_list_dir.replace('/', '\\')

# Fix mangled path
if not os.path.exists(images_list_dir) and 'config' in images_list_dir:
    if images_list_dir.startswith('C:\\Users\\'):
        config_part = images_list_dir.split('config')[1]
        images_list_dir = f"D:\\Foggy\\FogGuard\\config{config_part}"

print(f"Final path: {repr(images_list_dir)}")

# Check if file exists
if not os.path.exists(images_list_dir):
    print(f"Error: File {images_list_dir} does not exist!")
    exit(1)

with open(images_list_dir, 'r') as f:
    img_files = f.readlines()

# img_files = glob.glob(flags.directory + '/*.jpg')

psum, psum_sq, max_p, min_p = 0.0, 0.0, 0.0, float('inf')

midas.to(device)
midas.eval()

def normalize_depth_for_saving(depth_map):
    """
    Normalize depth map to 0-255 range for saving as PNG
    """
    # Convert to numpy if it's a tensor
    if torch.is_tensor(depth_map):
        depth_np = depth_map.cpu().numpy()
    else:
        depth_np = depth_map
    
    # Normalize to 0-255
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    
    if depth_max > depth_min:
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_normalized = np.zeros_like(depth_np)
    
    return depth_normalized.astype(np.uint8)

for i, filename in enumerate(tqdm(img_files)):
    filename = filename.strip().split()[0]  # Remove whitespace and take first part
    
    # Fix path for Windows
    if os.name == 'nt':
        filename = filename.replace('/', '\\')
    
    # Check if image file exists
    if not os.path.exists(filename):
        print(f"Warning: Image {filename} does not exist, skipping...")
        continue
    
    try:
        # img = np.array(Image.open(filename).convert('RGB'), dtype=np.uint8)
        img = cv2.imread(filename)
        if img is None:
            print(f"Warning: Could not read image {filename}, skipping...")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(image)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Create depth directory
        depth_parent = Path(Path(filename).parents[1], "depth")
        if not os.path.exists(depth_parent): 
            os.makedirs(depth_parent, exist_ok=True)

        # Get image name without extension
        image_name = Path(filename).stem
        depth_fn = Path(depth_parent, image_name + ".png")

        # Normalize depth for saving as PNG
        depth_normalized = normalize_depth_for_saving(prediction)
        
        # Save the normalized depth image
        success = cv2.imwrite(str(depth_fn), depth_normalized)
        if not success:
            print(f"Warning: Failed to save depth image for {filename}")
        
        # Also save raw depth as .pt file for later use
        depth_pt_fn = Path(depth_parent, image_name + ".pt")
        torch.save(prediction, depth_pt_fn)

        # Calculate statistics on original prediction
        psum += prediction.sum() / (prediction.shape[0] * prediction.shape[1])
        psum_sq += (prediction ** 2).sum() / (prediction.shape[0] * prediction.shape[1])

        max_p = max(max_p, prediction.max())
        min_p = min(min_p, prediction.min())
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

print("Statistics:")
print(f"Sum: {psum}")
print(f"Sum squared: {psum_sq}")

count = len(img_files)
# mean and STD
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = math.sqrt(total_var)

print(f"Mean: {total_mean}")
print(f"Std: {total_std}")
print(f"Max: {max_p}")
print(f"Min: {min_p}")

# output = prediction.cpu().numpy()

# plt.imshow(output)
# plt.show()