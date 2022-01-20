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
        self.transform = transform
        self.train_samples = [json.loads(x.rstrip()) for x in open(constants.TRAINING_ODGT_PATH, 'r')]
        self.val_samples = [json.loads(x.rstrip()) for x in open(constants.VALIDATION_ODGT_PATH, 'r')]

        (self.train_images, self.train_masks, self.val_images, self.val_masks) = self.load_data()

        self.train_length = 0
        self.val_length = 0

    def load_samples_with_walls(self, images:list, masks:list, purpose:str):
        """Loads on the images & masks lists all those samples containing walls.
        
        The purpose argument tells us if we're creating the dataset for "TRAINING"
        or "VALIDATION" purposes."""
        samples = self.train_samples if purpose == "TRAINING" else self.val_samples
        length = 0

        for sample in samples:
            # ADEChallengeData2016/images/<purpose>/ADE_train_00000001.jpg -> ADE_train_00000001
            image_name = sample["fpath_img"].split("/")[-1][:-4]

            # Only append the image if it contains a wall on it
            if self.SCENE_DICT[image_name] in self.SCENES_LIST:
                images.append(os.path.join(constants.DATA_FOLDER_NAME, sample['fpath_img']))
                masks.append(os.path.join(constants.DATA_FOLDER_NAME, sample["fpath_segm"]))
                length += 1

        if purpose == "TRAINING":
            self.train_length = length
        else:
            self.val_length = length


    def load_data(self):
        """Returns two lists containing the sorted paths to all images and masks containing walls."""
        # Loop through each image, check what scene it has and, if that scene is in the
        # scenes_list, add it to the list images and masks
        train_images, train_masks, val_images, val_masks = [], [], [], []

        self.load_samples_with_walls(train_images, train_masks, "TRAINING")
        self.load_samples_with_walls(val_images, val_masks, "VALIDATION") 

        return train_images, train_masks, val_images, val_masks
    
    def __len__(self):
        return self.train_length + self.val_length

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
    
    def preprocess(self, x:str, y:str):
        """Preprocess the input image and mask"""
        def f(x, y):
            # Paths have the following format b"path". We want to convert them
            # so they have the normal string format "".
            x = x.decode()
            y = y.decode()

            x = self.read_image(x)
            y = self.read_mask(y)

            x = x.long()
            y = y.long()

            return x, y

        
        image, mask = map(f, [x, y])
        image.reshape([self.SHAPE[0], self.SHAPE[1], 3])
        mask.reshape([self.SHAPE[0], self.SHAPE[1], 1])

        return image, mask
    
    def __getitem__(self, idx):
        img_path = self.train_images[idx] if idx < self.val_length else self.val_images[idx]
        mask_path = self.train_masks[idx] if idx < self.val_length else self.val_masks[idx]

        image, mask = self.preprocess(img_path, mask_path)

        if self.transform is not None:
            augmentations = self.transform(img=image, mask=mask)
            image, mask = augmentations["img"], augmentations["mask"]

        return image, mask

if __name__ == '__main__':
    wall_dataset = WallDataset()

    for i in range(3):
        image, mask = wall_dataset[i]
        print(f"Image shape: {image.shape}",
              f"Mask shape: {mask.shape}")