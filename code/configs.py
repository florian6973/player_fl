ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DATA_DIR = f'{ROOT_DIR}/data'
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import sys
sys.path.append(f'{ROOT_DIR}/code')
from torch.utils.data  import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR10
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from PIL import Image
from sklearn.preprocessing import StandardScaler
import albumentations 
from sklearn.model_selection import train_test_split
import copy
from collections import OrderedDict
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import torch.nn.functional as F
import pickle
import json
import time
import logging
from datetime import datetime
from functools import wraps
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import torch.multiprocessing as mp
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn import metrics
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import argparse 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 4

ALGORITHMS = [
    'local', 
    'fedavg', 
    'fedprox', 
    'pfedme', 
    'ditto', 
    'localadaptation', 
    'babu', 
    'fedlp', 
    'fedlama', 
    'pfedla', 
    'layerpfl',
    'layerpfl_minus_1',
    'layerpfl_plus_1'
]
DATASETS = [
    'FMNIST',
    'EMNIST',
    'CIFAR',
    'Sentiment',
    'ISIC',
    'mimic',
    'Heart'
]

DATASET_ALPHA = {
    'EMNIST':0.5,
    'CIFAR':0.5,
    "FMNIST":0.5
}

DEFAULT_PARAMS = {
    'FMNIST': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'num_clients': 5,
        'sizes_per_client': 2000,
        'classes': 10,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 20,
        'runs_lr': 3
    },
    'EMNIST': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'num_clients': 5,
        'sizes_per_client': 3000,
        'classes': 62,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 75,
        'runs': 10,
        'runs_lr': 3
    },
    'CIFAR': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4],
        'num_clients': 5,
        'sizes_per_client': 10000,
        'classes': 10,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 10,
        'runs_lr': 3
    },
    'ISIC': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'num_clients': 4,
        'sizes_per_client': None,
        'classes': 4,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 60,
        'runs': 3,
        'runs_lr': 1
    },
    'Sentiment': {
        'learning_rates_try': [1e-3, 5e-4, 1e-4, 8e-5],
        'num_clients': 15,
        'sizes_per_client': None,
        'classes': 2,
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 50,
        'runs': 10,
        'runs_lr': 3
    },
    'Heart': {
        'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2],
        'num_clients': 4,
        'sizes_per_client': None,
        'classes': 5,
        'batch_size': 32,
        'epochs_per_round': 1,
        'rounds': 20,
        'runs': 50,
        'runs_lr': 5
    },
    'mimic': {
        'learning_rates_try': [1e-3, 5e-4, 1e-4, 8e-5],
        'num_clients': 4,
        'sizes_per_client': None,
        'classes': 2,
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 25,
        'runs': 10,
        'runs_lr': 3
    }
}


LAYERS_TO_FEDERATE_DICT = {
    "layerpfl":{
            'EMNIST':['layer1.', 'layer2.', 'layer3.'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'],
            "Sentiment":['token_embedding_table1', 'position_embedding_table1', 'attention1', 'proj1'],
            "Heart": ['fc1', 'fc2'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1']
            },

    "babu":{
            'EMNIST':['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "Sentiment":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1'],
            "Heart": ['fc1', 'fc2', 'fc3'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1']
            },

    "layerpfl_minus_1":{
            'EMNIST':['layer1.', 'layer2.'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.'],
            "FMNIST":['layer1.', 'layer2.'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.'],
            "Sentiment":['token_embedding_table1', 'position_embedding_table1'],
            "Heart": ['fc1'],
            "mimic":['token_embedding_table1','position_embedding_table1']
            },

    "layerpfl_plus_1":{
            'EMNIST':['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "Sentiment":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1'],
            "Heart": ['fc1', 'fc2', 'fc3'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1']
            },
            
}

REG_PARAMS = {
    'fedprox': {
            'EMNIST': 0.1,
            'CIFAR': 0.1,
            "FMNIST":0.1,
            'ISIC':0.1,
            "Sentiment":0.1,
            "Heart": 0.1,
            "mimic":0.1
            },
    
    'pfedme': {
            'EMNIST': 0.1,
            'CIFAR': 0.1,
            "FMNIST":0.1,
            'ISIC':0.1,
            "Sentiment":0.1,
            "Heart": 0.1,
            "mimic":0.1
            },

    'ditto': {
            'EMNIST': 0.1,
            'CIFAR': 0.1,
            "FMNIST":0.1,
            'ISIC':0.1,
            "Sentiment":0.1,
            "Heart": 0.1,
            "mimic":0.1
            },
    
    'layerpfl': {
            'EMNIST': 0.1,
            'CIFAR': 0.1,
            "FMNIST":0.1,
            'ISIC':0.1,
            "Sentiment":0.1,
            "Heart": 0.1,
            "mimic":0.1
            },
    }

LAYER_PRESERVATION_RATES = {
    'EMNIST': 0.7,
    'CIFAR': 0.7,
    "FMNIST":0.7,
    'ISIC':0.7,
    "Sentiment":0.7,
    "Heart": 0.7,
    "mimic":0.7
}

HYPERNETWORK_PARAMS = {
    'embedding_dim': {
        'EMNIST': 32,
        'CIFAR': 64,
        "FMNIST":32,
        'ISIC':64,
        "Sentiment":64,
        "Heart": 16,
        "mimic":64
    },

    'hidden_dim': {
        'EMNIST': 64,
        'CIFAR': 128,
        "FMNIST":64,
        'ISIC':128,
        "Sentiment":128,
        "Heart": 16,
        "mimic":128
    },

    'hn_lr': {
        'EMNIST': 0.01,
        'CIFAR': 0.01,
        "FMNIST":0.01,
        'ISIC':0.01,
        "Sentiment":0.01,
        "Heart": 0.01,
        "mimic":0.01
    }
}

ATTENTION_MODELS = ['Sentiment', 'mimic']

