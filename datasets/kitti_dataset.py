import os
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import cv2
from matplotlib import colors
import glob
import albumentations as A
from dotmap import DotMap

class KITTI(Dataset):
    def __init__(self, dataset_params):
        self.dp = dataset_params
        with open(self.dp.pathfile, "r") as f:
            samples = [l.split() for l in f.readlines()]
            self.samples = []
            missing_img = []
            missing_depth= []
            for samp in samples:
                img_file = os.path.join(self.dp.img_path, samp[0])
                depth_file = os.path.join(self.dp.depth_path, samp[1])
                
                if os.path.isfile(img_file) and os.path.isfile(depth_file):
                    self.samples.append((img_file, depth_file))

        np.random.shuffle(self.samples)

        if self.dp.limit:
            self.samples = self.samples[:self.dp.limit]

        self.rc = A.RandomCrop(height=self.dp.input_shape[0], width=self.dp.input_shape[1])
        self.normalize_fn = A.Normalize()  # has defaults mean=(0.485, 0.456, 0.406) etc

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.3),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
            #               hue=0.2, always_apply=False, p=0.5),
            # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0,
            #                always_apply=False, p=0.5),
            # A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=19, always_apply=False, p=0.5),
            # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        ])

        self.np_to_tensor = torch.tensor
        # print(len(self.samples))

    def normalize(self, image):
        return self.normalize_fn(image=image)['image']
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, depth_file = self.samples[idx]
        image = Image.open(img_file)
        depth_gt = Image.open(depth_file)
        # print('depth_file', depth_file, type(depth_gt))

        # kb crop
        height = image.height
        width = image.width
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        x = np.array(image, dtype=np.float32)

        y = np.array(depth_gt, dtype=np.float32) / 256.0
        # print(len(np.unique(y)), print(y.shape))

        result = {}

        rc = self.rc(image=x, mask=y)
        result['prenorm_x'] = rc['image']
        result['x'] = self.np_to_tensor(self.normalize(rc['image'])).permute(2, 0, 1).float()
        result['y'] = self.np_to_tensor(rc['mask']).unsqueeze(0)

        if self.dp.is_train_dataset:
            augmented = self.aug(image=rc['image'], mask=rc['mask'])
            result['prenorm_x_aug'] = augmented['image']
            result['y_aug'] = self.np_to_tensor(augmented['mask']).unsqueeze(0)
            result['x_aug'] = self.np_to_tensor(
                self.normalize(augmented['image'])).permute(2, 0, 1).float()

        return result

if __name__=='__main__':
    kitti_train_params = DotMap()
    kitti_train_params.is_train_dataset = True
    kitti_train_params.batch_size = 16
    kitti_train_params.pathfile = '/home/khalil/Documents/depth/data/kitti/train.txt'
    kitti_train_params.input_shape = (352, 704, 3)
    kitti_train_params.img_path = '/home/khalil/Documents/depth/data/kitti/img'
    kitti_train_params.depth_path = '/home/khalil/Documents/depth/data/kitti/depth'
    dataset = KITTI(kitti_train_params)
    ds = dataset[0]
