"""
Store here the constants to be used on the dataset creation and on
the data loader.

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import os

DATA_FOLDER_NAME = "data"
TRAINING_ODGT_PATH = os.path.join(DATA_FOLDER_NAME, "training.odgt")
VALIDATION_ODGT_PATH = os.path.join(DATA_FOLDER_NAME, "validation.odgt")
SCENE_CATEGORIES_PATH = os.path.join(DATA_FOLDER_NAME,
                                    "ADEChallengeData2016",
                                    "sceneCategories.txt")
