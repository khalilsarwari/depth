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

class TODD(Dataset):
    def __init__(self, dataset_params):
        self.dp = dataset_params
        self.samples = []
        for location in os.listdir(self.dp.path):
            if location in self.dp.locations:
                loc_path = os.path.join(self.dp.path, location)
                for file in os.listdir(loc_path):
                    if 'depth' in file and file.endswith('.png'):
                        depth_file = os.path.join(loc_path, file)
                        img_file = depth_file.replace('_depth', '')
                        if os.path.isfile(img_file) and os.path.isfile(depth_file):
                            self.samples.append((img_file, depth_file))

        #np.random.shuffle(self.samples)
        self.samples = sorted(self.samples)
        print(self.samples[:10])

        if self.dp.limit:
            self.samples = self.samples[:self.dp.limit]

        self.rc = A.CenterCrop(height=self.dp.input_shape[0], width=self.dp.input_shape[1])
        self.normalize_fn = A.Normalize()  # has defaults mean=(0.485, 0.456, 0.406) etc

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])

        self.np_to_tensor = torch.tensor
        print("Found {} samples".format(len(self.samples)))
        

    def normalize(self, image):
        return self.normalize_fn(image=image)['image']
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, depth_file = self.samples[idx]
        image = Image.open(img_file)
        depth_gt = Image.open(depth_file)

        height = image.height
        width = image.width
        height = image.height
        width = image.width
        depth_gt = depth_gt.crop((50, 200, width-50, 200 + 352))
        image = image.crop((50, 200, width-50, 200 + 352))
        

        x = np.array(image, dtype=np.float32)
        # print('x shape', x.shape)

        y = (np.array(depth_gt, dtype=np.float32) / 100) # divide by 100 to convert cm to m
        # print(y.shape)
        # y should be in range 0 to 260 (meters)

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
    todd_train_params = DotMap()
    todd_train_params.is_train_dataset = True
    todd_train_params.batch_size = 16
    todd_train_params.input_shape = (352, 704, 3)
    todd_train_params.path = '/home/khalil/Documents/depth/data/todd/data'
    todd_train_params.locations = ['campbell','cupertino','losgatos', 'paloalto', 'saratoga']
    dataset = TODD(todd_train_params)
    ds = dataset[0]
    x = ds['prenorm_x']
    imx = Image.fromarray(x.astype(np.uint8))
    imx.save('test_x.png')
    y = ds['y'].permute(1, 2, 0) *255
    imy = Image.fromarray(y.numpy().squeeze().astype(np.uint8))
    imy.save('test_y.png')
