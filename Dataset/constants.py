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

# SCENES_LIST is a list with all the scenes of the ADE20K
# that can contain a wall
SCENES_LIST = ['bathroom', 'bedroom', 'kitchen', 'living_room',
               'art_gallery', 'art_studio', 'attic', 'auditorium',
                'shop', 'ballroom', 'bank_indoor', 'banquet_hall',
                'bar', 'basement', 'bookstore', 'childs_room',
                'classroom', 'room', 'closet', 'clothing_store',
                'computer_room', 'conference_room', 'corridor',
                'office', 'darkroom', 'dentists_office', 'diner_indoor',
                'dinette_home', 'dining_room', 'doorway_indoor',
                'dorm_room', 'dressing_room', 'entrance_hall', 'galley',
                'game_room', 'garage_indoor', 'gymnasium_indoor',
                'hallway', 'home_office', 'hospital_room', 'hotel_room',
                'jail_cell', 'kindergarden_classroom',
                'lecture_room', 'library_indoor', 'lobby', 'museum_indoor',
                'nursery', 'playroom', 'staircase',
                'television_studio', 'utility_room', 'waiting_room',
                'warehouse_indoor', 'youth_hostel']

