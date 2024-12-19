import torch
import torch.nn as nn
import losses as ls
import numpy as np
import random
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
from typing import Dict, Optional
from dataclasses import dataclass

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def set_seeds(seed_value=1):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainerConfig:
    """Configuration for training parameters."""
    dataset_name: str
    device: str
    learning_rate: float
    batch_size: int
    epochs: int = 5
    rounds: int = 20
    personalization_params: Optional[Dict] = None


LEARNING_RATES_TRY = {'EMNIST':[5e-3, 1e-3, 5e-4, 1e-4, 8e-5],
            'CIFAR':[5e-3, 1e-3, 5e-4, 1e-4],
            "FMNIST":[1e-3, 5e-4, 1e-4, 8e-5],
            "ISIC":[1e-3, 5e-3, 1e-4],
            "Sentiment":[1e-3, 5e-4, 1e-4,8e-5],
            "Heart":[5e-1, 1e-1, 5e-2, 1e-2, 5e-3],
            "mimic": [5e-4, 1e-4, 3e-4, 8e-5]}

NUM_SITES_DICT = {'EMNIST':5,
            'CIFAR':5,
            "FMNIST":5,
            "ISIC":4,
            "Sentiment": 15,
            "Heart":4,
            "mimic":4}

SIZES_PER_SITE_DICT = {'EMNIST':3000,
                'CIFAR':10000,
                "FMNIST":2000,
                "ISIC": None,
                "Sentiment": None,
                "Heart": None, 
                "mimic":None}


CLASSES_DICT = {'EMNIST':62,
            'CIFAR':10,
            "FMNIST":10,
            'ISIC':4,
            "Sentiment":2,
            "Heart":5,
            "mimic":2}

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

LR_DICT_ALG = {'EMNIST':{'Single':(1e-3,None), 'FedAvg':(1e-3,None), 'LayerPFL':(1e-3,1e-3),'FedProx':(5e-3, None), 
                         'pFedMe':(5e-2, None), 'Ditto':(1e-3, None), 'LocalAdaptation':(1e-3, None), 'BABU':(1e-3,0),
                         'LayerPFL_minus_1':(5e-4,5e-4), 'LayerPFL_plus_1':(1e-3,1e-3), 'FedLP':(5e-4,5e-4),
                         'FedLAMA':(1e-3,1e-3), 'pFedLA': (5e-3,5e-3)},
            'CIFAR':{'Single':(1e-3,None), 'FedAvg':(1e-3,None), 'LayerPFL':(1e-3,1e-3),'FedProx':(1e-3, None), 
                     'pFedMe':(5e-2, None), 'Ditto':(1e-3, None), 'LocalAdaptation':(1e-3, None), 'BABU':(5e-4,0),
                     'LayerPFL_minus_1':(1e-3,1e-3), 'LayerPFL_plus_1':(5e-4,5e-4), 'FedLP':(1e-3,1e-3),
                    'FedLAMA':(5e-4,5e-4), 'pFedLA': (1e-3,1e-3)},
            "FMNIST":{'Single':(1e-3,None), 'FedAvg':(5e-4,None), 'LayerPFL':(1e-3,1e-3),'FedProx':(5e-4, None), 
                      'pFedMe':(5e-2, None), 'Ditto':(5e-4, None), 'LocalAdaptation':(5e-4, None), 'BABU':(1e-3,0),
                      'LayerPFL_minus_1':(1e-3,1e-3), 'LayerPFL_plus_1':(1e-3,1e-3), 'FedLP':(1e-3,1e-3),
                         'FedLAMA':(1e-3,1e-3), 'pFedLA': (1e-3,1e-3)},
            "ISIC":{'Single':(1e-3,None), 'FedAvg':(1e-3,None), 'LayerPFL':(5e-4,5e-4),'FedProx':(1e-3, None), 
                    'pFedMe':(5e-3, None), 'Ditto':(1e-3, None), 'LocalAdaptation':(1e-3, None), 'BABU':(1e-3,0),
                    'LayerPFL_minus_1':(5e-4,5e-4), 'LayerPFL_plus_1':(1e-3,1e-3), 'FedLP':(5e-3,5e-3),
                         'FedLAMA':(1e-3,1e-3), 'pFedLA': (1e-3,1e-3)},
            "Sentiment":{'Single':(5e-4,None), 'FedAvg':(1e-3,None), 'LayerPFL':(8e-5,8e-5),'FedProx':(1e-3, None), 
                         'pFedMe':(1e-2, None), 'Ditto':(1e-3, None), 'LocalAdaptation':(1e-3, None), 'BABU':(8e-5,0),
                         'LayerPFL_minus_1':(8e-5,8e-5), 'LayerPFL_plus_1':(1e-4,1e-4), 'FedLP':(5e-4,5e-4),
                         'FedLAMA':(1e-3,1e-3), 'pFedLA': (1e-3,1e-3)},
            "Heart":{'Single':(5e-2,None), 'FedAvg':(1e-1,None), 'LayerPFL':(1e-2,1e-2),'FedProx':(5e-2, None),
                      'pFedMe':(1e-1, None), 'Ditto':(5e-2, None), 'LocalAdaptation':(1e-2, None), 'BABU':(1e-1,0),
                      'LayerPFL_minus_1':(5e-2,5e-2), 'LayerPFL_plus_1':(5e-2,5e-2), 'FedLP':(5e-2,5e-2), 
                         'FedLAMA':(5e-1,5e-1), 'pFedLA': (1e-2,1e-2)},
            "mimic":{'Single':(8e-5,None), 'FedAvg':(5e-4,None), 'LayerPFL':(3e-4,3e-4),'FedProx':(5e-4, None),
                     'pFedMe':(1e-3, None), 'Ditto':(5e-4, None), 'LocalAdaptation':(5e-4, None), 'BABU':(5e-4,0),
                     'LayerPFL_minus_1':(8e-5,8e-5), 'LayerPFL_plus_1':(5e-4,5e-4), 'FedLP':(3e-4,3e-4),
                         'FedLAMA':(8e-5,8e-5), 'pFedLA': (5e-4,5e-4)}}

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

LAYERS_TO_FEDERATE_DICT = {
    "LayerPFL":{'EMNIST':['layer1.', 'layer2.', 'layer3.'],
                'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'],
                "FMNIST":['layer1.', 'layer2.', 'layer3.'],
                'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'],
                "Sentiment":['token_embedding_table1', 'position_embedding_table1', 'attention1', 'proj1'],
                "Heart": ['fc1', 'fc2'],
                "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1']},

    "BABU":{'EMNIST':['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "Sentiment":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1'],
            "Heart": ['fc1', 'fc2', 'fc3'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1']},
    
    "LayerPFL_minus_1":{'EMNIST':['layer1.', 'layer2.'],
                'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.'],
                "FMNIST":['layer1.', 'layer2.'],
                'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.'],
                "Sentiment":['token_embedding_table1', 'position_embedding_table1'],
                "Heart": ['fc1'],
                "mimic":['token_embedding_table1','position_embedding_table1']},
 
    "LayerPFL_plus_1":{'EMNIST':['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.', 'fc1'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
            "Sentiment":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1'],
            "Heart": ['fc1', 'fc2', 'fc3'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1']},

    "FedLP":{'EMNIST':['layer1.', 'layer2.', 'layer3.', 'fc1', 'fc2'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1', 'fc2'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.', 'fc1', 'fc2'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1', 'fc2'],
            "Sentiment":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1', 'fc2'],
            "Heart": ['fc1', 'fc2', 'fc3', 'fc4'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1', 'fc2']},

        }

ATTENTION_MODELS = ['Sentiment', 'mimic']

CLIP_GRAD = []

RUNS =  {'EMNIST':10,
        'CIFAR':10,
        "FMNIST":10,
        "ISIC":3,
        "Sentiment": 20,
        "Heart": 50,
        "mimic":10}

ALGORITHMS = ['Single','FedAvg','FedProx','pFedMe', 'Ditto', 'LocalAdaptation', 'BABU', 
              'LayerPFL', 'LayerPFL_minus_1', 'LayerPFL_plus_1', 'FedLP', 'FedLAMA']

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
