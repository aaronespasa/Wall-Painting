"""
Create a DataLoader for PyTorch using the ADE20K dataset.

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

