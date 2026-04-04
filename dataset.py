import glob
import os
import random
import torch
import numpy as np
import cv2

'''
SyntheticHomographyDataset
Generates synthetic training pairs for homography estimation from the HPatches dataset.
Each sample consists of a grayscale image patch (img0), a randomly warped version (img1),
and the ground-truth homography matrix (H_gt) that maps img0 to img1.

Parameters:
  num_pairs  - Virtual dataset size (number of samples per epoch). Each sample is generated
               on-the-fly by randomly selecting an image and applying a random homography,
               so this controls how many iterations one epoch contains, not actual stored pairs.
  patch_size - Height and width (in pixels) of the square patch cropped from each image.
               Larger values preserve more context but require more GPU memory.

Usage:
  dataset = SyntheticHomographyDataset(image_dir="path/to/hpatches", num_pairs=10000, patch_size=480)
  loader  = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
  for img0, img1, H_gt in loader:
      # img0, img1: (B, 1, H, W) float tensors normalized to [0, 1]
      # H_gt:       (B, 3, 3)  ground-truth homography matrices

HPatches directory structure expected: <image_dir>/<scene>/1.ppm

How to call this class: import sys; sys.path.append('..'); from dataset import SyntheticHomographyDataset; dataset = SyntheticHomographyDataset(image_dir="path/to/hpatches", num_pairs=10000, patch_size=480)
'''

class SyntheticHomographyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, num_pairs=10000, patch_size=480):
        self.images = glob.glob(os.path.join(image_dir, "*/1.ppm"))
        self.num_pairs = num_pairs
        self.patch_size = patch_size

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # randomly choose an image from hpatches dataset
        img_path = random.choice(self.images)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # randomly crop a patch from the image
        h, w = img.shape
        y = random.randint(0, max(0, h - self.patch_size))
        x = random.randint(0, max(0, w - self.patch_size))
        patch = img[y:y+self.patch_size, x:x+self.patch_size]
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))

        # apply random homography to the patch
        corners = np.array([[0,0],[self.patch_size,0],
                           [self.patch_size,self.patch_size],
                           [0,self.patch_size]], dtype=np.float32)
        perturb = np.random.uniform(-40, 40, (4, 2)).astype(np.float32)
        H = cv2.getPerspectiveTransform(corners, corners + perturb)
        warped = cv2.warpPerspective(patch, H, (self.patch_size, self.patch_size))

        # transform to tensors
        img0 = torch.from_numpy(patch).float() / 255.0
        img1 = torch.from_numpy(warped).float() / 255.0
        H_gt = torch.from_numpy(H).float()

        return img0.unsqueeze(0), img1.unsqueeze(0), H_gt