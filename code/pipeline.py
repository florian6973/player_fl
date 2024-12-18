ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import torch
import sys
sys.path.append(f'{ROOT_DIR}/code')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import dataset_processing as dp
import trainers as tr
import configs as cs
import pickle
import gc
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import models as mod
cs.set_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINER_DICT = {'Single':tr.SingleTrainer,
                'FedAvg':tr.FederatedTrainer, 
                'FedProx':tr.FedProxTrainer, 
                'pFedMe':tr.pFedMEModelTrainer, 
                'Ditto':tr.DittoModelTrainer,
                "LocalAdaptation":tr.LocalAdaptationTrainer,
                "BABU": tr.BabuTrainer,
                'LayerPFL':tr.LayerPFLTrainer,
                'LayerPFL_minus_1':tr.LayerPFLTrainer,
                'LayerPFL_plus_1':tr.LayerPFLTrainer,
                'FedLP': tr.FedLPTrainer,
                'FedLAMA': tr.FedLAMATrainer,
                'pFedLA': tr.pFedLATrainer}

def save_results(dataset, results, results_site, losses, cross_device = False):
    if cross_device:
        with open(f'{ROOT_DIR}/results/{dataset}_results_cd.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{ROOT_DIR}/results/{dataset}_results_site_cd.pkl', 'wb') as f:
            pickle.dump(results_site, f)
        with open(f'{ROOT_DIR}/results/{dataset}_losses_cds.pkl', 'wb') as f:
            pickle.dump(losses, f)
    else:
        with open(f'{ROOT_DIR}/results/{dataset}_results_pfedla.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{ROOT_DIR}/results/{dataset}_results_site_pfedla.pkl', 'wb') as f:
            pickle.dump(results_site, f)
        with open(f'{ROOT_DIR}/results/{dataset}_losses_pfedla.pkl', 'wb') as f:
            pickle.dump(losses, f)
    return

def run_algorithm(algorithm, dataset, cross_device = False):
    """ Function to wrap the training of 1 algorithm together """
    #load data
    dataloader = cs.DATA_DICT[dataset]
    natural_partition = cs.PARTITION_DICT[dataset]
    SIZE_PER_SITE = cs.SIZES_PER_SITE_DICT[dataset]
    NUM_SITES = cs.NUM_SITES_DICT[dataset]
    if natural_partition:
        #for data collected at different sites already
        sites_data = [i for i in range(NUM_SITES)]
    else:
        #split the data using dirichlet distribution of labels
        sites_data = dp.create_datasets_per_site(dataloader, SIZE_PER_SITE, NUM_SITES, cs.DATASET_ALPHA[dataset])
    sites_loaders = {}
    for site, site_data in enumerate(sites_data):
        sites_loaders[site] = cs.DATALOADER_DICT[dataset](site_data)
    
    
    results = {}
    results_site = {}
    losses = {}
    
    with tqdm(total=len(range(cs.RUNS[dataset])), desc=f"Training {algorithm} on {dataset}") as pbar:
        for run in range(1,cs.RUNS[dataset]+1):
            cs.set_seeds(run)
            try:
                NUM_SITES = cs.NUM_SITES_DICT[dataset]
                CLASSES = cs.CLASSES_DICT[dataset]
                if algorithm == 'pFedLA':
                    data_model = cs.MODEL_DICT[dataset](CLASSES)
                    weight_layer_names = [name for name, _ in data_model.named_parameters()]
                    hypernetwork = mod.HyperNetwork(embedding_dim = 128, hidden_dim = 1000, layers = weight_layer_names , num_clients = len(sites_data))
                    model  = (data_model, hypernetwork)
                else:
                    model = cs.MODEL_DICT[dataset](CLASSES)
                LRs = cs.LR_DICT_ALG[dataset][algorithm]
                epochs = cs.EPOCHS_DICT[dataset]
                loss_fct = cs.LOSSES_DICT[dataset]
                if dataset in cs.FOCAL_LOSS_DATASETS:
                    loss_fct = loss_fct(CLASSES, alpha = cs.LOSS_ALPHA[dataset]) # instantiate with correct number of classes and alpha
                if algorithm in ['LayerPFL', 'BABU', 'LayerPFL_plus_1', 'LayerPFL_minus_1', 'FedLP']:
                    layers_to_federate = cs.LAYERS_TO_FEDERATE_DICT[algorithm][dataset]
                else:
                    layers_to_federate = None
                trainer = TRAINER_DICT[algorithm](device, dataset, NUM_SITES, CLASSES, layers_to_federate, epochs, batch_size = cs.BATCH_SIZE_DICT[dataset], cross_device = cross_device)
                results[run], results_site[run], losses[run] = trainer.run_pipeline(sites_loaders, model, loss_fct, LRs, clear = True)
                del trainer
                clear_data()
            except Exception as e:
                print(f"Error during run {run} for algorithm {algorithm}: {e}")
            pbar.update(1)
            print('\n')
    return results, results_site, losses

def run_dataset_pipeline(dataset, cross_device = False):
    #Run training
    if (f'{dataset}_results.pkl' in os.listdir(f'{ROOT_DIR}/results')) and not cross_device:
        with open(f'{ROOT_DIR}/results/{dataset}_results.pkl', 'rb') as f:
            results = pickle.load(f)
        with open(f'{ROOT_DIR}/results/{dataset}_results_site.pkl', 'rb') as f:
            results_site = pickle.load(f)
        with open(f'{ROOT_DIR}/results/{dataset}_losses.pkl', 'rb') as f:
            losses = pickle.load(f)
        ALGORITHMS = ['pFedLA'] #list(set(cs.ALGORITHMS)- set(list(results.keys())))
    
    elif cross_device:
        results = {}
        results_site = {}
        losses = {}
        ALGORITHMS = ['FedAvg','FedProx','pFedMe', 'Ditto', 'LocalAdaptation', 'BABU', 
                    'LayerPFL']
    else:
        results = {}
        results_site = {}
        losses = {}
        ALGORITHMS = cs.ALGORITHMS
        
                          
    with tqdm(total=len(ALGORITHMS), desc=f"Training Algorithms on {dataset}") as pbar:
        for algorithm in ALGORITHMS:
            results[algorithm], results_site[algorithm], losses[algorithm] = run_algorithm(algorithm, dataset, cross_device)
            pbar.update(1)
            print('\n')

    #Save
    save_results(dataset, results, results_site, losses, cross_device)
    return

def clear_data():
    gc.collect()
    torch.cuda.empty_cache()

def main(datasets, cross_device):
    for dataset in datasets:
        run_dataset_pipeline(dataset, cross_device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--datasets")
    parser.add_argument("-cd", "--cross_device", action="store_true", default = False)
    args = parser.parse_args()
    datasets = args.datasets
    cross_device = args.cross_device
    datasets = datasets.split(',')
    
    main(datasets, cross_device)