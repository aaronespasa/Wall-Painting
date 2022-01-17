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
    
    def __getitem__(self, idx):
        """To be implemented..."""
        ...
