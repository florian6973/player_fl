ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import torch
import sys
sys.path.append(f'{ROOT_DIR}/code')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import dataset_processing as dp
import models as mod
import trainers as tr
import configs as cs
import pickle
import gc
import configs as cs
import argparse
import torch.multiprocessing as mp
import os
from tqdm import tqdm
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

# Range of LRs to try
LR_TUNING_LOOP = {'EMNIST':[5e-3, 1e-3, 5e-4, 1e-4, 8e-5],
            'CIFAR':[5e-3, 1e-3, 5e-4, 1e-4],
            "FMNIST":[1e-3, 5e-4, 1e-4, 8e-5],
            "ISIC":[1e-3, 5e-3, 1e-4],
            "Sentiment":[1e-3, 5e-4, 1e-4,8e-5],
            "Heart":[5e-1, 1e-1, 5e-2, 1e-2, 5e-3],
            "mimic": [5e-4, 1e-4, 3e-4, 8e-5]}

#Separate training LRs needed as different optimizer
LR_TUNING_LOOP_PFEDME = {'EMNIST':[5e-2, 1e-2, 5e-3, 1e-3],
            'CIFAR':[5e-1, 1e-1, 5e-2, 1e-2],
            "FMNIST":[5e-1, 1e-1, 5e-2],
            "ISIC":[1e-2, 5e-3, 1e-3],
            "Sentiment":[1e-1, 5e-2, 1e-2, 1e-3],
            "Heart":[5e-1, 1e-1, 5e-2, 1e-2, 5e-3],
            "mimic": [5e-1, 1e-1, 5e-2]}

# Create LR tuning dict
LR_TUNING_DICT = {}
for dataset in cs.LR_DICT_ALG:
    LR_TUNING_DICT[dataset] = {}
    for alg in cs.LR_DICT_ALG[dataset]:
        if cs.LR_DICT_ALG[dataset][alg][-1] is None:
            LR_TUNING_DICT[dataset][alg] = [(LR, None) for LR in (LR_TUNING_LOOP_PFEDME[dataset] if alg == 'pFedMe' else LR_TUNING_LOOP[dataset])]

        else:
            LR_TUNING_DICT[dataset][alg] = [(LR, LR) for LR in (LR_TUNING_LOOP_PFEDME[dataset] if alg == 'pFedMe' else LR_TUNING_LOOP[dataset])]

# Runs per dataset and lr
RUNS =  {'EMNIST':1,
        'CIFAR':1,
        "FMNIST":1,
        "ISIC":1,
        "Sentiment": 3,
        "Heart": 5,
        "mimic":3}


def run_algorithm(algorithm, dataset, multi):
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
    with tqdm(total=len(LR_TUNING_DICT[dataset][algorithm])*RUNS[dataset], desc=f"Training {algorithm} on {dataset}") as pbar:
        for LRs in LR_TUNING_DICT[dataset][algorithm]:
            results[LRs[0]] = {}
            losses[LRs[0]] = {}
            results_site[LRs[0]] = {}
            for run in range(1,RUNS[dataset]+1):
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
                    epochs = cs.EPOCHS_DICT[dataset]
                    loss_fct = cs.LOSSES_DICT[dataset]
                    if dataset in cs.FOCAL_LOSS_DATASETS:
                        loss_fct = loss_fct(CLASSES, alpha = cs.LOSS_ALPHA[dataset]) # instantiate with correct number of classes and alpha
                    if algorithm in ['LayerPFL', 'BABU', 'LayerPFL_plus_1', 'LayerPFL_minus_1', 'FedLP']:
                        layers_to_federate = cs.LAYERS_TO_FEDERATE_DICT[algorithm][dataset]
                    else:
                        layers_to_federate = None
                    trainer = TRAINER_DICT[algorithm](device, dataset, NUM_SITES, CLASSES, layers_to_federate, epochs, batch_size = cs.BATCH_SIZE_DICT[dataset])
                    results[LRs[0]][run], results_site[LRs[0]][run], losses[LRs[0]][run] = trainer.run_pipeline(sites_loaders, model, loss_fct, LRs, clear = True)
                    del trainer
                    clear_data()
                except Exception as e:
                    print(f"Error during run {run} for algorithm {algorithm}: {e}", flush = True)
            
                pbar.update(1)
                print('\n')
    return results, results_site, losses

def save_results(dataset, results, results_site, losses, multi, workbook = False):
    if multi:
        results_dir = 'results_multi'
    else:
        results_dir = 'results'
    if workbook:
        with open(f'{ROOT_DIR}/{results_dir}/{dataset}_results_lr_wb.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{ROOT_DIR}/{results_dir}/{dataset}_results_site_lr_wb.pkl', 'wb') as f:
            pickle.dump(results_site, f)
        with open(f'{ROOT_DIR}/{results_dir}/{dataset}_losses_lr_wb.pkl', 'wb') as f:
            pickle.dump(losses, f)
    else:
        with open(f'{ROOT_DIR}/{results_dir}/{dataset}_results_lr_2.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(f'{ROOT_DIR}/{results_dir}/{dataset}_results_site_lr_2.pkl', 'wb') as f:
            pickle.dump(results_site, f)
        with open(f'{ROOT_DIR}/{results_dir}/{dataset}_losses_lr_2.pkl', 'wb') as f:
            pickle.dump(losses, f)
    return

def run_dataset_pipeline(dataset, multi, workbook = False):
    #Run training
    if multi:
        results_dir = 'results_multi'
    else:
        results_dir = 'results'
    if f'{dataset}_results_lr.pkl' in os.listdir(f'{ROOT_DIR}/{results_dir}'):
        with open(f'{ROOT_DIR}/results/{dataset}_results_lr.pkl', 'rb') as f:
            results = pickle.load(f)
        with open(f'{ROOT_DIR}/results/{dataset}_results_site_lr.pkl', 'rb') as f:
            results_site = pickle.load(f)
        with open(f'{ROOT_DIR}/results/{dataset}_losses_lr.pkl', 'rb') as f:
            losses = pickle.load(f)
        ALGORITHMS = ['FedLP'] #list(set(cs.ALGORITHMS)- set(list(results.keys())))
    else:
        results = {}
        results_site = {}
        losses = {}
        ALGORITHMS = cs.ALGORITHMS
                          
    with tqdm(total=len(ALGORITHMS), desc=f"Training Algorithms on {dataset}") as pbar:
        for algorithm in ALGORITHMS:
            results[algorithm], results_site[algorithm], losses[algorithm] = run_algorithm(algorithm, dataset, multi)
            pbar.update(1)
            print('\n')
    #Save
    save_results(dataset, results, results_site, losses, workbook)
    return

def clear_data():
    gc.collect()
    torch.cuda.empty_cache()

def main(datasets, multi):
    for dataset in datasets:
        run_dataset_pipeline(dataset, multi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--datasets")
    parser.add_argument('--multi', action='store_true', default=False,
                      help='Multi-client (default: False)')
    parser.add_argument('--no-multi', action='store_false', dest='multi',
                      help='Multi-client')
    args = parser.parse_args()
    datasets = args.datasets
    datasets = datasets.split(',')
    multi = args.multi

    
    main(datasets, multi)