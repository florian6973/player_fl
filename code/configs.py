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
    num_clients: int = 5
    personalization_params: Optional[Dict] = None

def get_parameters_for_dataset(DATASET):
    default_params = {
        'EMNIST': {
            'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4, 8e-5],
            'num_clients': 5,
            'sizes_per_site': 3000,
            'classes': 62,
            'batch_size': 128,
            'rounds': 10,
            'runs': 50,
            'runs_lr': 5
        },
        'CIFAR': {
            'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
            'num_clients': 5,
            'sizes_per_site': 10000,
            'classes': 10,
            'batch_size': 128,
            'rounds': 10,
            'runs': 20,
            'runs_lr': 5
        },
        'FMNIST': {
            'learning_rates_try': [1e-3, 5e-4, 1e-4, 8e-5],
            'num_clients': 5,
            'sizes_per_site': 2000,
            'classes': 10,
            'batch_size': 128,
            'rounds': 10,
            'runs': 50,
            'runs_lr': 5
        },
        'ISIC': {
            'learning_rates_try': [1e-3, 5e-3, 1e-4],
            'num_clients': 4,
            'sizes_per_site': None,
            'classes': 4,
            'batch_size': 32,
            'rounds': 10,
            'runs': 3,
            'runs_lr': 1
        },
        'Sentiment': {
            'learning_rates_try': [1e-3, 5e-4, 1e-4, 8e-5],
            'num_clients': 15,
            'sizes_per_site': None,
            'classes': 2,
            'batch_size': 128,
            'rounds': 10,
            'runs': 10,
            'runs_lr': 3
        },
        'Heart': {
            'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2, 5e-3],
            'num_clients': 4,
            'sizes_per_site': None,
            'classes': 5,
            'batch_size': 128,
            'rounds': 10,
            'runs': 50,
            'runs_lr': 5
        },
        'mimic': {
            'learning_rates_try': [5e-4, 1e-4, 3e-4, 8e-5],
            'num_clients': 4,
            'sizes_per_site': None,
            'classes': 2,
            'batch_size': 128,
            'rounds': 10,
            'runs': 10,
            'runs_lr': 3
        }
    }

    params = default_params.get(DATASET)
    if not params:
        raise ValueError(f"Dataset {DATASET} is not supported.")

    return params

def get_personalization_config(server_type, dataset_name):
    """Get personalization parameters for specific server type and dataset."""
    params = {}
    
    # Layer-based methods
    if server_type in ['layerpfl', 'babu', 'layerpfl_minus_1', 'layerpfl_plus_1']:
        params['layers_to_include'] = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]
    
    # FedLP specific parameters
    elif server_type == 'fedlp':
        params['layer_preserving_rate'] = LAYER_PRESERVATION_RATES[dataset_name]
    
    # Regularization-based methods
    elif server_type in ['fedprox', 'pfedme', 'ditto']:
        params['reg_param'] = REG_PARAMS[server_type][dataset_name]
    
    # FedLAMA specific parameters
    elif server_type == 'fedlama':
        params['tau_prime'] = 2
        params['phi'] = 2
    
    # pFedLA specific parameters
    elif server_type == 'pfedla':
        params['embedding_dim'] = HYPERNETWORK_PARAMS['embedding_dim'][dataset_name]
        params['hidden_dim'] = HYPERNETWORK_PARAMS['hidden_dim'][dataset_name]
        params['hn_lr'] = HYPERNETWORK_PARAMS['hn_lr'][dataset_name]

    return params


CLASSES_DICT = {'EMNIST':62,
            'CIFAR':10,
            "FMNIST":10,
            'ISIC':4,
            "Sentiment":2,
            "Heart":5,
            "mimic":2}


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
                'CIFAR': 0.15,
                "FMNIST":0.1,
                'ISIC':0.1,
                "Sentiment":0.1,
                "Heart": 0.1,
                "mimic":0.1
                },
        
        'pfedme': {
                'EMNIST': 0.1,
                'CIFAR': 0.15,
                "FMNIST":0.1,
                'ISIC':0.1,
                "Sentiment":0.1,
                "Heart": 0.1,
                "mimic":0.1
                },

        'ditto': {
                'EMNIST': 0.1,
                'CIFAR': 0.15,
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
