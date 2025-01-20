import torch
import torch.nn as nn
import dataset_processing as dp
import models as mod
import losses as ls
import numpy as np
import random
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR10

def set_seeds(seed_value=1):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SIZES_PER_SITE_DICT = {'EMNIST':3000,
                'CIFAR':10000,
                "FMNIST":2000,
                "ISIC": None,
                "Sentiment": None,
                "Heart": None, 
                "mimic":None}

NUM_SITES_DICT = {'EMNIST':5,
            'CIFAR':5,
            "FMNIST":5,
            "ISIC":4,
            "Sentiment": 15,
            "Heart":4,
            "mimic":4}

CLASSES_DICT = {'EMNIST':62,
            'CIFAR':10,
            "FMNIST":10,
            'ISIC':4,
            "Sentiment":2,
            "Heart":5,
            "mimic":2}

DATALOADER_DICT = {'EMNIST':dp.EMNISTDataset,
            'CIFAR':dp.CIFARDataset,
            "FMNIST":dp.FMNISTDataset,
            "ISIC":dp.ISICDataset,
            "Sentiment": dp.SentimentDataset,
            "Heart": dp.HeartDataset,
            "mimic":dp.mimicDataset}

MODEL_DICT = {'EMNIST':mod.EMNIST,
            'CIFAR':mod.CIFAR,
            "FMNIST":mod.FMNIST,
            "ISIC": mod.ISIC,
            "Sentiment": mod.Sentiment,
            "Heart": mod.Heart,
            "mimic":mod.mimic}


BATCH_SIZE_DICT = {'EMNIST':128,
                'CIFAR':128,
                "FMNIST":128,
                "ISIC": 32,
                "Sentiment": 128,
                "Heart": 128, 
                "mimic":128}

LR_DICT = {'EMNIST':5e-4,
            'CIFAR':5e-4,
            "FMNIST":7e-4,
            "ISIC":7e-4,
            "Sentiment":13e-5,
            "Heart":75e-4,
            "mimic": 9e-4}


LOSSES_DICT = {'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            "FMNIST": nn.CrossEntropyLoss(),
            "ISIC": ls.MulticlassFocalLoss,
            "Sentiment": nn.CrossEntropyLoss(),
            "Heart": ls.MulticlassFocalLoss,
            "mimic": ls.MulticlassFocalLoss}

FOCAL_LOSS_DATASETS = ['Heart', 'ISIC', 'mimic']
LOSS_ALPHA = {"Heart": [0.12939189, 0.18108108, 0.22331081, 0.22364865, 0.24256757],
              "ISIC": [0.87868852, 0.88131148, 0.82793443, 0.41206557],
             "mimic": [0.15,0.85]}
LOSS_GAMMA = {"Heart": 3,
              "ISIC": 1,
             "mimic": 1}

EPOCHS_DICT = {'EMNIST':75,
            'CIFAR':50,
            "FMNIST":75,
            "ISIC":50,
            "Sentiment": 75,
            "Heart": 50,
            "mimic":50}

ATTENTION_MODELS = ['Sentiment', 'mimic']

CLIP_GRAD = []

RUNS =  {'EMNIST':10,
        'CIFAR':10,
        "FMNIST":10,
        "ISIC":3,
        "Sentiment": 20,
        "Heart": 50,
        "mimic":10}


DATASET_ALPHA = {'EMNIST':0.5,
                'CIFAR':0.5,
                "FMNIST":0.5}

PARTITION_DICT = {'EMNIST': False,
                    'CIFAR':False,
                    "FMNIST":False,
                    "ISIC":True,
                    "Sentiment": True,
                    "Heart": True,
                    "mimic":True}

DATA_DICT = {'EMNIST': EMNIST(f'{ROOT_DIR}/data/EMNIST', split = 'byclass', download = False),
                    'CIFAR':CIFAR10(f'{ROOT_DIR}/data/CIFAR10', download = False),
                    "FMNIST":FashionMNIST(f'{ROOT_DIR}/data/FMNIST', download = False),
                    "ISIC":None,
                    "Sentiment": None,
                    "Heart": None,
                    "mimic":None}
