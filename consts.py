import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from IPython.display import Image as DisplayImage
from tqdm.notebook import tqdm




IS_COLAB = False
base_dir = Path('/content/drive/My Drive/Colab Notebooks/Mask Detector') if IS_COLAB else Path('.')
models_dir = base_dir / 'models'
resnet_model_dir = models_dir / 'resnet'
haarcascade_model_dir = models_dir / 'haarcascade'
datasets_dir = base_dir / 'datasets'
videos_dir = base_dir / 'videos'
images_dir = base_dir / 'images'
gradcam_evaluation_dir = base_dir / 'gradcam_evaluation'
masked_gradcam_evaluation_dir = gradcam_evaluation_dir / 'masked'
no_mask_gradcam_evaluation_dir = gradcam_evaluation_dir / 'no mask'
rmfd_dataset_dir = datasets_dir / 'self-built-masked-face-recognition-dataset'
imgs_w_mask = rmfd_dataset_dir / 'AFDB_masked_face_dataset'
imgs_wo_mask = rmfd_dataset_dir / 'AFDB_face_dataset'
ALREADY_TRAINED = True
device = 'cpu'
seed = 0