import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from models.my_models import my_models
from torchvision import transforms
import torchvision
import pickle
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from my_modules import trainer
import dnnlib
import legacy
import re
from torch_utils import gen_utils
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

