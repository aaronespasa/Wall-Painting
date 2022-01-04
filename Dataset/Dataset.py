"""
Download and organize the ADE20K dataset on a folder called "data".

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import tensorflow as tf

from glob import glob
import os
import json
from urllib.request import urlretrieve
from zipfile import ZipFile

from PIL import Image
import cv2

import matplotlib.pyplot as plt
import numpy as np

class Dataset:
    def __init__(self):
        self.DATA_FOLDER_NAME = "data"
        self.TRAINING_ODGT_PATH = os.path.join(self.DATA_FOLDER_NAME, "training.odgt")
        self.VALIDATION_ODGT_PATH = os.path.join(self.DATA_FOLDER_NAME, "validation.odgt")

        self.train_samples = [json.loads(x.rstrip()) for x in open(self.TRAINING_ODGT_PATH, 'r')]
        self.val_samples = [json.loads(x.rstrip()) for x in open(self.VALIDATION_ODGT_PATH, 'r')]

        self.SCENE_CATEGORIES_PATH = os.path.join(self.DATA_FOLDER_NAME, "ADEChallengeData2016", "sceneCategories.txt")

