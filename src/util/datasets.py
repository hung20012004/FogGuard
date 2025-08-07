# import torch.nn.functional as F
import torchvision.transforms.functional as F
import torch
import glob
import shutil
import math
import random
import os
from tqdm import tqdm, trange
import xml.etree.ElementTree as ET
import pathlib
from pathlib import Path
import pandas as pd
import warnings
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, WeightedRandomSampler, SequentialSampler
# import torch.utils.data
import torchvision
import PIL.Image
from PIL import Image
from PIL import ImageFile
# import transforms
import util.transforms as utransforms
from torchvision import transforms
from torchvision.io import read_image
# import utils as utils
import util.utils as utils
from util.logger import Logger
# from logger import Logger

ImageFile.LOAD_TRUNCATED_IMAGES = True


class yolo_dataset(Dataset):
    def __init__(self, data_config, data_type, transform=None, image_size=416, 
             batch_size=16, num_samples=None):
        import os
        
        self.classes = utils.load_classes(data_config["names"])
        self.data_type = data_type
        
        # Fix path handling for Windows
        image_dir_path = data_config[data_type]
        print(f"Original {data_type} path: {repr(image_dir_path)}")
        
        # Handle Windows path issues
        if os.name == 'nt':  # Windows
            image_dir_path = image_dir_path.replace('/', '\\')
        
        # Fix mangled path (same logic as utils.py)
        if not os.path.exists(image_dir_path) and 'config' in image_dir_path:
            if image_dir_path.startswith('C:\\Users\\'):
                config_part = image_dir_path.split('config')[1]
                image_dir_path = f"D:\\Foggy\\FogGuard\\data\\VOC\\config{config_part}"
        
        print(f"Final {data_type} path: {repr(image_dir_path)}")
        self.image_dir_path = image_dir_path
        
        self.transform = transform
        self.img_size = image_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.multiscale = False
        self.batch_count = 0
        self.name = data_config["dataset"]
        
        with open(self.image_dir_path, 'r') as f:
            self.img_files = f.readlines()
        
        if num_samples is None:
            # self.img_files = self.img_files[:num_samples]
            self.num_samples = len(self.img_files)
        
        self.label_files = []
        for path in self.img_files:
            # Strip whitespace and newlines from path
            path = path.strip()
            
            image_dir = os.path.dirname(path)
            label_dir = f"labels-{len(self.classes)}".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)
    def __getitem__(self, index):
        import os
        from pathlib import Path
        import warnings
        import numpy as np
        from PIL import Image
        
        #  Image
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            
            # Fix path for Windows
            if os.name == 'nt':  # Windows
                img_path = os.path.normpath(img_path)
            
            print(f"Trying to read image: {repr(img_path)}")
            
            # Check if file exists
            if not os.path.exists(img_path):
                print(f"Image file does not exist: {img_path}")
                return None
                
            img = Image.open(img_path).convert('RGB')  # Keep as PIL Image for transform
            
            # load depth channel
            depth_parent = Path(Path(img_path).parents[1], "depth")
            
            # Fix image name extraction for Windows paths
            if os.name == 'nt':
                image_name = os.path.basename(img_path)[:-4]  # Remove .jpg extension
            else:
                image_name = img_path.split("/")[-1][:-4]
                
            depth_fn = Path(depth_parent, image_name + '.png')
            
            print(f"Looking for depth file: {depth_fn}")
            
            # Check if depth file exists
            if os.path.exists(depth_fn):
                depth = read_image(depth_fn)
            else:
                print(f"Depth file not found: {depth_fn}")
                # Create dummy depth data if depth file doesn't exist
                height, width = img.size[1], img.size[0]  # PIL Image uses (width, height)
                depth = np.zeros((1, height, width), dtype=np.uint8)
                
        except Exception as e:
            print(f"Could not read image '{img_path}': {e}")
            return None
            
        #  Label
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            
            # Fix label path for Windows
            if os.name == 'nt':
                label_path = os.path.normpath(label_path)
            
            print(f"Trying to read label: {repr(label_path)}")
            
            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"Label file does not exist: {label_path}")
                # Create empty label if file doesn't exist
                boxes = np.empty((0, 5))
            else:
                # Ignore warning if file is empty
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    boxes = np.loadtxt(label_path).reshape(-1, 5)
                    
        except Exception as e:
            print(f"Could not read label '{label_path}': {e}")
            # Create empty label on error
            boxes = np.empty((0, 5))
            
        #  Transform
        if self.transform:
            try:
                # Convert PIL to numpy for transform
                img_np = np.array(img, dtype=np.uint8)
                img_transformed, bb_targets = self.transform((img_np, boxes))
                
                if hasattr(self, 'transform_depth') and depth is not None:
                    if len(depth.shape) == 3:
                        depth = self.transform_depth(depth[0])
                    else:
                        depth = self.transform_depth(depth)
                else:
                    # Simple depth transform if method doesn't exist
                    pass
            except Exception as e:
                print(f"Could not apply transform: {e}")
                # Don't return None, create a valid output
                img_transformed = np.array(img, dtype=np.uint8)
                bb_targets = boxes
        else:
            img_transformed = np.array(img, dtype=np.uint8)
            bb_targets = boxes
                
        # return img_path, img, bb_targets
        return img_path, img_transformed, depth, bb_targets
    def collate_fn(self, batch, eval_flag=False):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        # paths, imgs, bb_targets = list(zip(*batch))
        paths, imgs, depth, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = [torch.from_numpy(target).float() if isinstance(target, np.ndarray) else target for target in bb_targets]
        bb_targets = torch.cat(bb_targets, 0)

        # return paths, imgs, bb_targets
        depth = torch.stack([self.transform_depth(img) for img in depth])
        return paths, imgs, depth, bb_targets

    def __len__(self):
        return len(self.img_files)

    def make_folders(self, path):
        """Create directory if not exists"""
        if not os.path.exists(path):
            # shutil.rmtree(path)
            os.makedirs(path)

    def replace_folders(self, path):
        """Replace the directory if exists"""
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def create_dataloader(self):
        # sampler = BatchSampler(RandomSampler(dataset, replacement=True),
        #                          self.batch_size, drop_last=True)
        if self.data_type == "train":
            train_ratio = 0.8
            train_size = int(train_ratio * len(self))
            valid_size = len(self) - train_size
            train_ds, valid_ds = torch.utils.data.random_split(self,
                                                            [train_size, valid_size])

            if self.name == 'voc':
                train_sampler = RandomSampler(train_ds, replacement=True,
                                              num_samples=int(train_ratio *\
                                                        self.num_samples))

                valid_sampler = RandomSampler(valid_ds, replacement=True,
                                            num_samples=int((1 - train_ratio) *\
                                                            self.num_samples))

            elif self.name == 'voc-rtts':
                train_sampler = RandomSampler(train_ds, replacement=True,
                                            num_samples=int((1 - train_ratio) *\
                                                            self.num_samples))
                                        # num_samples=self.num_samples)
                valid_sampler = RandomSampler(valid_ds, replacement=True,
                                            num_samples=int((1 - train_ratio) *\
                                                            self.num_samples))
                                            # num_samples=self.num_samples)
                # if self.data_type == "test":
                #     rtts_frag = 432.0 / 5385.0
                # else:
                #     rtts_frag = 3890.0 / 20442.0

                # rtts_rate = .05
                # weights = [rtts_rate] * int(rtts_frag * len(self)) +\
                #             [1 - rtts_rate] * int((1 - rtts_frag) * len(self))
                # sampler = WeightedRandomSampler(weights, self.num_samples)

            else:
                sampler = SequentialSampler(self)

            train_dataloader = DataLoader(
                train_ds,
                sampler=train_sampler,
                batch_size=self.batch_size,
                # shuffle=self.shuffle,
                # num_workers=n_cpu,
                pin_memory=False,
                collate_fn=self.collate_fn,
                worker_init_fn=utils.worker_seed_set
            )
            val_dataloader = DataLoader(
                valid_ds,
                sampler=valid_sampler,
                batch_size=self.batch_size,
                pin_memory=False,
                collate_fn=self.collate_fn,
                worker_init_fn=utils.worker_seed_set
            )
            # else:
            #     dataloader = DataLoader(
            #         self,
            #         batch_size=self.batch_size,
            #         shuffle=False,
            #         # num_workers=n_cpu,
            #         pin_memory=True,
            #         collate_fn=self.collate_fn,
            #         worker_init_fn=utils.worker_seed_set
            #     )

            return train_dataloader, val_dataloader



        elif self.data_type == "test":
            test_sampler = RandomSampler(self, replacement=False)
            test_dataloader = DataLoader(
                self,
                sampler=test_sampler,
                batch_size=self.batch_size,
                pin_memory=False,
                collate_fn=self.collate_fn,
                worker_init_fn=utils.worker_seed_set
            )
            return test_dataloader

        else:
            exit("Wrong dataset type")
 
    def transform_depth(self, img):
        # Since depth files are missing, create dummy depth tensor directly
        # Return tensor với shape phù hợp: (C, H, W)
        dummy_depth = torch.zeros(1, self.img_size, self.img_size)
        return dummy_depth
 

class SquarePadSize:
    def __init__(self, img_size):
        self.img_size = img_size
        "docstring"

    def __call__(self, image):
        # print('before: ', image.min(), image.max())
        s = image.shape
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        padding = (hp, vp, hp, vp)
        image = F.pad(image, padding, fill=0, padding_mode='constant')
        # padded = torch.nn.functional.pad(image, padding, 0, mode='constant')
        image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0),
                                            size=(self.img_size,self.img_size),
                                            mode="bicubic").squeeze().squeeze()
        image = torch.clamp(image, 0.)
        # print("after:", image.min(), image.max())
        return image
        # max_wh = max(image.size)
        # p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        # p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        # padding = (p_left, p_top, p_right, p_bottom)
        # return F.pad(image, padding, 0, 'constant')


def pad_to_square(img, pad_value=0):
    # c, h, w = img.shape
    h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Đảm bảo image có đúng format cho interpolate
    # Nếu image là (H, W, C) thì chuyển thành (C, H, W)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    # Resize image
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), 
        size=size, 
        mode="nearest"
    ).squeeze(0)
    
    return image

def _create_data_loader(img_path, batch_size, img_size, n_cpu, transform,
                        num_samples=None, multiscale_training=False,
                        shuffle=True):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=transform,
        num_samples=num_samples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=utils.worker_seed_set)
    return dataloader


def show(sample):

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()




def main():
    args = utils.get_parsed_args()
    logger = Logger(args.logdir)  # Tensorboard logger
    data_config = utils.parse_data_config(args.data)
    class_names = utils.load_classes(data_config["names"])
    device = utils.get_device(args)

    batch_size = 4
    image_size = 256
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Transforms images to a PyTorch Tensor
    transform = transforms.simple_transform(image_size, norm_mean, norm_std)

    # voc train dataset
    train_ds = VOC(data_config["train"], data_config["dest"], "train",
                   class_names, transform, image_size, batch_size).create_dataloader()
    test_ds = VOC(data_config["test"], data_config["dest"], "test",
                  class_names, transform, image_size, batch_size).create_dataloader()


if __name__ == '__main__':
    main()
