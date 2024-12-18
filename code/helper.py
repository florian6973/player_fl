ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DATA_DIR = f'{ROOT_DIR}/data'
import torch
import pandas as pd
import numpy as np
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR10
import os

class UnifiedDataLoader:
    """
    Unified data loader that handles multiple data formats and prepares them
    for the DataPartitioner and DataPreprocessor pipeline
    """
    def __init__(self, root_dir: str, dataset_name: str):
        self.root_dir = root_dir
        self.data_dir = f'{root_dir}/data'
        self.dataset_name = dataset_name
        
    def load(self):
        """
        Load data and convert to a standardized DataFrame format with 'data', 'label', and 'site' columns
        """
        if self.dataset_name in ['EMNIST', 'CIFAR', 'FMNIST']:
            return self._load_benchmark_images()
        elif self.dataset_name == 'ISIC':
            return self._load_isic()
        elif self.dataset_name == 'Sentiment':
            return self._load_sentiment()
        elif self.dataset_name == 'MIMIC':
            return self._load_mimic()
        elif self.dataset_name == 'Heart':
            return self._load_heart()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_benchmark_images(self):
        """Handle torchvision datasets"""
        dataset_classes = {
            'EMNIST': lambda: EMNIST(f'{self.data_dir}/EMNIST', split='byclass', download=False),
            'CIFAR': lambda: CIFAR10(f'{self.data_dir}/CIFAR10', download=False),
            'FMNIST': lambda: FashionMNIST(f'{self.data_dir}/FMNIST', download=False)
        }
        
        dataset = dataset_classes[self.dataset_name]()
        data = dataset.data.numpy()
        labels = dataset.targets.numpy()
        
        # Assign sites randomly for benchmark datasets
        sites = [None for _ in range(len(labels))]
        
        return pd.DataFrame({
            'data': list(data),
            'label': labels,
            'site': sites
        })

    def _load_isic(self):
        """Handle ISIC image dataset"""
        all_data = []
        
        for site in range(6):
            file_path = f'{self.data_dir}/{self.dataset_name}/site_{site}_metadata.csv'
            files = pd.read_csv(file_path)
            image_files = [f'{self.data_dir}/ISIC_2019_Training_Input_preprocessed/{file}.jpg' 
                          for file in files['image']]
            
            site_df = pd.DataFrame({
                'data': image_files,
                'label': files['label'].values,
                'site': site
            })
            all_data.append(site_df)
            
        return pd.concat(all_data, ignore_index=True)

    def _load_sentiment(self):
        """Handle Sentiment dataset with tensor dictionaries"""
        all_data = []
        
        for device in range(15):
            file_path = f'{self.data_dir}/Sentiment/data_device_{device}_indices.pth'
            site_data = torch.load(file_path)
            
            site_df = pd.DataFrame({
                'data': list(site_data['data'].numpy()),
                'label': site_data['label'].numpy(),
                'mask': list(site_data['mask'].numpy()),
                'site': device
            })
            all_data.append(site_df)
            
        return pd.concat(all_data, ignore_index=True)

    def _load_mimic(self):
        """Handle MIMIC dataset with tensor dictionaries"""
        all_data = []
        diagnoses = ['mi', 'gi', 'infection', 'brain']
        
        for i, dx in enumerate(diagnoses):
            file_path = f'{self.data_dir}/mimic_iii/dataset_concatenated_{dx}_indices.pt'
            site_data = torch.load(file_path)
            
            site_df = pd.DataFrame({
                'data': list(site_data['data'].numpy()),
                'label': site_data['label'].numpy(),
                'mask': list(site_data['mask'].numpy()),
                'site': i
            })
            all_data.append(site_df)
            
        return pd.concat(all_data, ignore_index=True)

    def _load_heart(self):
        """Handle Heart dataset from CSV files"""
        columns = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'sugar', 'ecg', 'max_hr', 'exercise_angina',
            'exercise_ST_depression', 'target'
        ]
        
        all_data = []
        sites = ['cleveland', 'hungarian', 'switzerland', 'va']
        
        for i, site in enumerate(sites):
            file_path = f'{self.data_dir}/{self.dataset_name}/processed.{site}.data'
            site_data = pd.read_csv(
                file_path,
                names=columns,
                na_values='?',
                usecols=columns
            ).dropna()
            
            site_data['site'] = i
            all_data.append(site_data)
            
        data = pd.concat(all_data, ignore_index=True)
        
        # Convert features to numpy arrays for consistency
        feature_cols = [col for col in columns if col != 'target']
        data['data'] = list(data[feature_cols].values)
        data['label'] = data['target']
        
        return data[['data', 'label', 'site']]