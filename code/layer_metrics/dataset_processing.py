ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import torch
import pandas as pd
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/helper')
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from PIL import Image
from sklearn.preprocessing import StandardScaler

#DATALOADERS
class EMNISTDataset(Dataset):
    def __init__(self, data):
        self.images = data[0]
        self.labels = data[1]
        self.transform = self._transform()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = transforms.ToPILImage()(image)
        image = self.transform(image)
        return image, label

    def _transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform
    
class CIFARDataset(Dataset):
    def __init__(self, data):
        self.images = data[0]
        self.labels = data[1]
        self.transform = self._transform()

    def __getitem__(self, index):
        img = self.images[index]
        target = self.labels[index]
        img = self.transform(img)
        return img, target

    def _transform(self):
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
    def __len__(self):
        return len(self.images)

class FMNISTDataset(Dataset):
    def __init__(self, data):
        self.images = data[0]
        self.labels = data[1]
        self.transform = self._transform()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = transforms.ToPILImage()(image)
        image = self.transform(image)
        return image, label

    def _transform(self):
        transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform

class ISICDataset(Dataset):
    def __init__(self, site):
        """ metadata = pd.DataFrame()
        for site in range(4):
            m = pd.read_csv(f'{ROOT_DIR}/data/ISIC/site_{site}_metadata.csv')
            metadata = pd.concat([metadata, m]) """
        metadata = pd.read_csv(f'{ROOT_DIR}/data/ISIC/site_{site}_metadata.csv')
        self.data = list(metadata['path'].values)
        self.labels = list(metadata['target'].values)
        self.transform = CustomTransform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype = torch.int64)
        return image, label

class CustomTransform:
    def __init__(self):
        self.sz = 200
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.transforms = transforms.Compose([
            transforms.CenterCrop(self.sz),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __call__(self, img):
        img = self.transforms(img)
        return img


class SentimentDataset(Dataset):
    def __init__(self, device):
        file_path = f'{ROOT_DIR}/data/Sentiment/data_device_{device}_indices.pth'
        loaded_data = torch.load(file_path) 
        self.data = loaded_data['data']
        self.labels = loaded_data['labels']
        self.masks = loaded_data['masks']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        mask = self.masks[idx]
        return (x, mask), y
    
class mimicDataset(Dataset):
    def __init__(self, site, mortality = True):
        site_mapper = {0:'mi', 1:'gi', 2:'infection', 3:'brain'}
        dx = site_mapper[site]
        file_path = f'{ROOT_DIR}/data/mimic_iii/dataset_concatenated_{dx}_indices.pt'
        loaded_data = torch.load(file_path) 
        self.data = loaded_data['data']
        self.labels = loaded_data['labels']['Mortality'] if mortality else loaded_data['labels']['LOS'] 
        self.masks = loaded_data['masks']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        mask = self.masks[idx]
        return (x, mask), y

class HeartDataset(Dataset):
    def __init__(self, site):
        #data loading
        columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'slope_ST', 'number_major_vessels', 'thalassemia_hx', 'target']
        used_columns =  ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression', 'target']
        SITE_DICT = {0:'cleveland', 1: "hungarian", 2:"switzerland", 3:"va"}
        data = pd.read_csv(f'{ROOT_DIR}/data/Heart/processed.{SITE_DICT[site]}.data', names = columns, na_values='?', usecols = used_columns).dropna()
        
        #assign data and labels
        self.data = data.iloc[:,:-1]
        self._transform()
        self.labels = list(data.iloc[:,-1].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx, :], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype = torch.long )
        return features, label

    def _transform(self):
        cols_to_scale = ['age', 'chest_pain_type', 'resting_bp', 'cholesterol', 'ecg', 'max_hr',  'exercise_ST_depression']
        #Set scaling params
        scaler = StandardScaler()
        mean = np.array([ 53.0972973 ,   3.22702703, 132.75405405, 220.13648649,
                            0.63513514, 138.74459459,   0.89432432])
        var = np.array([7.02459463e+01, 8.16756772e-01, 3.45293057e+02, 4.88330934e+03,
                         5.92069868e-01, 5.29172208e+02, 1.11317517e+00])
        scaler.mean_ = mean
        scaler.var_ = var
        scaler.scale_ = np.sqrt(var) 

        #scale
        self.data[cols_to_scale] = scaler.transform(self.data[cols_to_scale])
        self.data = self.data.values
        return


def split_labels_dirichlet(NUM_SITES, data, labels, alpha, mask=None):
    """ Label splitting by dirichlet distribution across variable number of site. To be used when no natural partition """
    unique_labels, counts = np.unique(labels, return_counts=True)
    alpha_values = np.ones(NUM_SITES) * alpha
    label_weights_per_site = np.zeros((len(unique_labels), NUM_SITES))
    
    for idx, count in enumerate(counts):
        label_weights = np.random.dirichlet(alpha_values) * count
        label_weights_per_site[idx] = label_weights

    if mask is not None:
        site_data = [([], [], []) for _ in range(NUM_SITES)]
    else:
        site_data = [([], []) for _ in range(NUM_SITES)] 

    for label, weight in zip(unique_labels, label_weights_per_site):
        indices_for_label = np.where(labels == label)[0]
        np.random.shuffle(indices_for_label) 
        start = 0
        for site in range(NUM_SITES):
            end = start + int(weight[site])
            site_data[site][0].extend(data[indices_for_label[start:end]])
            site_labels = [label.item() for label in labels[indices_for_label[start:end]]]
            site_data[site][1].extend(site_labels)
            if mask is not None:
                site_data[site][2].extend(mask[indices_for_label[start:end]])
                
            start = end
            
    return site_data

def create_datasets_per_site(dataloader, SIZE_PER_SITE, NUM_SITES, alpha):
    """ Create the dataset per site: To be used when no natural partition """
    data_full, labels_full = dataloader.data, dataloader.targets
    mask_full = getattr(dataloader, 'masks', None)
    if isinstance(labels_full, list):
        labels_full= torch.tensor(labels_full)
    if SIZE_PER_SITE is not None:
        SIZE = SIZE_PER_SITE*NUM_SITES
        data, labels = data_full[:SIZE], labels_full[:SIZE]
        if mask_full is not None:
            mask = mask_full[:SIZE]
        else:
            mask = mask_full
    else:
        data, labels = data_full, labels_full
        mask = mask_full

    sites_data = split_labels_dirichlet(NUM_SITES, data, labels, alpha, mask)
    return sites_data

