"""
Download and organize the ADE20K dataset on a folder called "data".

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import os
import json
import progressbar
from urllib.request import urlretrieve
from zipfile import ZipFile

import importlib.util
CONSTANTS_PATH = os.path.join(os.path.dirname(__file__), 'constants.py')
spec = importlib.util.spec_from_file_location('constants', CONSTANTS_PATH)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

DATA_FOLDER_NAME = constants.DATA_FOLDER_NAME
SCENE_CATEGORIES_PATH = constants.SCENE_CATEGORIES_PATH
SCENES_LIST = constants.SCENES_LIST
TRAINING_ODGT_PATH = constants.TRAINING_ODGT_PATH
VALIDATION_ODGT_PATH = constants.VALIDATION_ODGT_PATH

pbar = None

class ADE20K:
    """
    Base class for the ADE20K dataset.
    Downloads the dataset and creates the necessary files for this dataset.
    """
    def __init__(self):
        self.ABSOLUTE_PATH = os.path.dirname(__file__)

        self.DATA_FOLDER_NAME = os.path.join(self.ABSOLUTE_PATH, DATA_FOLDER_NAME)
        self.TRAINING_ODGT_PATH = os.path.join(self.ABSOLUTE_PATH, TRAINING_ODGT_PATH)
        self.VALIDATION_ODGT_PATH = os.path.join(self.ABSOLUTE_PATH, VALIDATION_ODGT_PATH)

        # Create the data folder, download the ADE20K dataset and download the ODGT files
        self.create_folders()

        self.train_samples = [json.loads(x.rstrip()) for x in open(self.TRAINING_ODGT_PATH, 'r')]
        self.val_samples = [json.loads(x.rstrip()) for x in open(self.VALIDATION_ODGT_PATH, 'r')]
        
        self.SCENE_CATEGORIES_PATH = os.path.join(self.ABSOLUTE_PATH, SCENE_CATEGORIES_PATH)

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
            urlretrieve(train_val_url, train_val_file_name, self.show_progress)
            print("Downloaded succesfully!")

            print("Downloading testing data...")
            urlretrieve(test_url, test_file_name, self.show_progress)
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
            urlretrieve(TRAINING_ODGT_URL, self.TRAINING_ODGT_PATH, self.show_progress)
            print(f"The file training.odgt has been downloaded succesfully!")

        # validation.odgt
        if os.path.isfile(self.VALIDATION_ODGT_PATH):
            print(f"The file validation.odgt already exists!")
        else:
            VALIDATION_ODGT_URL = "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/validation.odgt"
            urlretrieve(VALIDATION_ODGT_URL, self.VALIDATION_ODGT_PATH, self.show_progress)
            print(f"The file validation.odgt has been downloaded succesfully!")

    def show_progress(self, block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None


    ##########################################################
    ##########################################################

    def create_folders(self):
        self.create_data_folder()
        self.download_ade20k_dataset()
        self.download_odgts()

if __name__ == '__main__':
    dataset = ADE20K()
    dataset.create_folders()
