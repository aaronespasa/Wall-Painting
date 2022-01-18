"""
Create a DataLoader for PyTorch using the ADE20K dataset.

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import os
import json
from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
import constants

class WallDataset(Dataset):
    def __init__(self, transform=None):
        self.train_samples = [json.loads(x.rstrip()) for x in open(constants.TRAINING_ODGT_PATH, 'r')]
        self.val_samples = [json.loads(x.rstrip()) for x in open(constants.VALIDATION_ODGT_PATH, 'r')]
        
        (self.train_images, self.train_masks, self.val_images, self.val_masks) = self.load_data()

        self.length = 0

    def load_data(self):
        """Returns two lists containing the sorted paths to all images and masks.
        
        It just returns those images which contain walls."""
        # Loop through each image, check what scene it has and, if that scene is in the
        # scenes_list, add it to the list images and masks
        train_images = []
        train_masks = []
        val_images = []
        val_masks = []

        for sample in self.train_samples:
            # ADEChallengeData2016/images/training/ADE_train_00000001.jpg -> ADE_train_00000001
            image_name = sample['fpath_img'][37:55]

            # Only append the image if it contains a wall in it
            if self.SCENE_DICT[image_name] in self.SCENES_LIST:
                train_images.append(os.path.join("data", sample['fpath_img']))
                train_masks.append(os.path.join("data", sample['fpath_segm']))
                self.length += 1

        for sample in self.val_samples:
            # ADEChallengeData2016/images/validation/ADE_val_00000006.jpg -> ADE_val_00000006
            image_name = sample['fpath_img'][39:55]

            # Only append the image if it contains a wall in it
            if self.SCENE_DICT[image_name] in self.SCENES_LIST:
                val_images.append(os.path.join("data", sample['fpath_img']))
                val_masks.append(os.path.join("data", sample['fpath_segm']))
                self.length += 1

        return train_images, train_masks, val_images, val_masks
    
    def __len__(self):
        return self.length

    def read_image(self, img_path: str):
        """Resizes and normalizes an image located at img_path"""
        x = cv2.imread(img_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.SHAPE[0], self.SHAPE[1]))
        x = x / 255.0
        x = x.astype(np.float32)
        return x


    def read_mask(self, mask_path: str):
        """Convert a mask into grayscale, resizes it and normalizes it"""
        x = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (self.SHAPE[0], self.SHAPE[1]))

        x = np.array(x, dtype=np.int16) - 1
        x[x > 0] = 1

        x = x / 255.0
        # Images have dimensions (self.SHAPE[0], self.SHAPE[1], 3).
        # However, masks have dimensions (self.SHAPE[0], self.SHAPE[1]).
        # Thus, we need to add a dimensions so they have the following shape: self.SHAPE[0], self.SHAPE[1], 1).
        x = np.expand_dims(x, axis=-1)  # axis 2 = axis -1 = last dimension
        x = x.astype(np.float32)
        return x
    
    def __getitem__(self, idx):
        """To be implemented..."""
        ...
