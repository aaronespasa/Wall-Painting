"""
Download and organize the ADE20K dataset on a folder called "data".

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import constants
import os
import json
from urllib.request import urlretrieve
from zipfile import ZipFile

class ADE20K:
    """
    Base class for the ADE20K dataset.
    Downloads the dataset and creates the necessary files for this dataset.
    """
    def __init__(self):
        self.DATA_FOLDER_NAME = "data"
        self.TRAINING_ODGT_PATH = os.path.join(self.DATA_FOLDER_NAME, "training.odgt")
        self.VALIDATION_ODGT_PATH = os.path.join(self.DATA_FOLDER_NAME, "validation.odgt")

        # Create the data folder, download the ADE20K dataset and download the ODGT files
        self.create_folders()

        self.train_samples = [json.loads(x.rstrip()) for x in open(self.TRAINING_ODGT_PATH, 'r')]
        self.val_samples = [json.loads(x.rstrip()) for x in open(self.VALIDATION_ODGT_PATH, 'r')]

        self.SCENE_CATEGORIES_PATH = os.path.join(self.DATA_FOLDER_NAME, "ADEChallengeData2016", "sceneCategories.txt")
        self.SCENE_DICT = self.build_scene_dict()

    def build_scene_dict(self):
        """
        Example of the return dictionary {Image-name, Scene-on-the-image}:
        {'ADE_train_00000001': airport_terminal,
         'ADE_train_00000051': bathroom, ...}
        """
        if os.path.isfile(self.SCENE_CATEGORIES_PATH):
            scene_dict = {}
            val_count = 0
            train_count = 0

            with open(self.SCENE_CATEGORIES_PATH, 'r') as scene_file:
                for line in scene_file:
                    img_name, scene_name = line.split(' ')
                    # scene_name = 'airport_terminal\n' -> scene_name[:-1] = 'airport_terminal'
                    scene_name = scene_name[:-1]
                    scene_dict[img_name] = scene_name

            return scene_dict
            
        raise FileNotFoundError(
            "sceneCategories.txt file does not exist. Make sure you first execute Dataset.create_folders()")

    ##########################################################    
    ######################## DOWNLOADS #######################

    def create_data_folder(self):
        """Create the folder called data/"""
        if not(self.DATA_FOLDER_NAME in os.listdir()):
            os.mkdir(self.DATA_FOLDER_NAME)
            print(f"The folder {self.DATA_FOLDER_NAME}/ has been created!")
        else:
            print(f"The folder {self.DATA_FOLDER_NAME}/ already exists!")
    
    def download_ade20k_dataset(self):
        """
        Download the ADE20K dataset from the following website:
        http://sceneparsing.csail.mit.edu/

        All the data is stored on the folder "data".
        """
        train_val_file_name = "ade20k-train-val.zip"
        test_file_name = "ade20k-test.zip"

        if ("ADEChallengeData2016" in os.listdir(self.DATA_FOLDER_NAME)
                and "release_test" in os.listdir(self.DATA_FOLDER_NAME)):
            print("The dataset ADE20K has already been downloaded!")
        else:
            train_val_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
            test_url = "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip"

            print("Downloading training and validation data...")
            urlretrieve(url=train_val_url, filename=train_val_file_name)
            print("Downloaded succesfully!")

            print("Downloading testing data...")
            urlretrieve(url=test_url, filename=test_file_name)
            print("Downloaded succesfully!")

            print(f"Extracting the content of both zips on the {self.DATA_FOLDER_NAME} folder...")
            zip_ref = ZipFile(file=train_val_file_name, mode="r")
            zip_ref.extractall(self.DATA_FOLDER_NAME)
            zip_ref.close()

            zip_ref = ZipFile(file=test_file_name, mode="r")
            zip_ref.extractall(self.DATA_FOLDER_NAME)
            zip_ref.close()

            # Delete the zip files
            os.remove(train_val_file_name)
            os.remove(test_file_name)

    def download_odgts(self):
        """
        Download the training.odgt and the validation.odgt.

        Both are json files containing the images with their respective annotations, widht and height.
        
        Example of training.odgt:
        [{
            'fpath_img': 'ADEChallengeData2016/images/training/ADE_train_00000001.jpg',
            'fpath_segm': 'ADEChallengeData2016/annotations/training/ADE_train_00000001.png',
            'width': 683,
            'height': 512
        }, ...]
        """
        # training.odgt
        if os.path.isfile(self.TRAINING_ODGT_PATH):
            print(f"The file training.odgt already exists!")
        else:
            TRAINING_ODGT_URL = "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/training.odgt"
            urlretrieve(url=TRAINING_ODGT_URL, filename=self.TRAINING_ODGT_PATH)
            print(f"The file training.odgt has been downloaded succesfully!")

        # validation.odgt
        if os.path.isfile(self.VALIDATION_ODGT_PATH):
            print(f"The file validation.odgt already exists!")
        else:
            VALIDATION_ODGT_URL = "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/validation.odgt"
            urlretrieve(url=VALIDATION_ODGT_URL, filename=self.VALIDATION_ODGT_PATH)
            print(f"The file validation.odgt has been downloaded succesfully!")

    ##########################################################
    ##########################################################

    def create_folders(self):
        self.create_data_folder()
        self.download_ade20k_dataset()
        self.download_odgts()
