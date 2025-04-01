ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from configs import *

def set_seeds(seed_value=1):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(seed_value=1)

def get_parameters_for_dataset(DATASET):
    params = DEFAULT_PARAMS.get(DATASET)
    if not params:
        raise ValueError(f"Dataset {DATASET} is not supported.")
    return params

def get_algorithm_config(server_type, dataset_name):
    """Get algorithm specific parameters for specific server type and dataset."""
    params = {}
    
    # Layer-based methods
    if server_type == 'babu':
        params['layers_to_include'] = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]
    
    if server_type == 'layerpfl':
        params['layers_to_include'] = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]
        params['reg_param'] = REG_PARAMS['layerpfl'][dataset_name]  # Optional, not currently used

    elif server_type == 'layerpfl_random':
        # Randomly select a prefix of layers, ensuring it's not identical to either:
        # - the fixed `layerpfl` configuration
        # - the full list of layers (i.e., all layers federated)
        fixed_layers = LAYERS_TO_FEDERATE_DICT['layerpfl'][dataset_name]
        random_layers = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]
        
        while True:
            idx = np.random.randint(len(random_layers))
            selected_layers = random_layers[:idx+1] # 0-based indexing and we include up until the selected layer
            if (selected_layers != fixed_layers) and (selected_layers != random_layers):
                break

        params['layers_to_include'] = selected_layers
        
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

def cleanup_gpu():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def move_to_device(batch, device):
    """Move batch data to device, handling both single tensors and lists/tuples of tensors."""
    if isinstance(batch, (list, tuple)):
        return [x.to(device) if x is not None else None for x in batch]
    return batch.to(device)


class ExperimentType:
    LEARNING_RATE = 'learning_rate'
    EVALUATION = 'evaluation'

class ExperimentConfig:
    def __init__(self, dataset, experiment_type):
        self.dataset = dataset
        self.experiment_type = experiment_type

class ResultsManager:
    def __init__(self, root_dir, dataset, experiment_type):
        self.root_dir = root_dir
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.results_structure = {
            ExperimentType.LEARNING_RATE: {
                'directory': 'lr_tuning',
                'filename_template': f'{dataset}_lr_tuning.pkl',
            },
            ExperimentType.EVALUATION: {
                'directory': 'evaluation',
                'filename_template': f'{dataset}_evaluation.pkl'
            },
        }

    def _get_results_path(self, experiment_type):
        experiment_info = self.results_structure[experiment_type]
        return os.path.join(RESULTS_DIR,
                            experiment_info['directory'], 
                            experiment_info['filename_template'])

    def load_results(self, experiment_type):
        path = self._get_results_path(experiment_type)

        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_results(self, results, experiment_type):
        path = self._get_results_path(experiment_type)
        with open(path, 'wb') as f:
            pickle.dump(results, f)

    def append_or_create_metric_lists(self, existing_dict, new_dict):
        if existing_dict is None:
            return {k: [v] if not isinstance(v, dict) else 
                   self.append_or_create_metric_lists(None, v)
                   for k, v in new_dict.items()}
        
        for key, new_value in new_dict.items():
            if isinstance(new_value, dict):
                if key not in existing_dict:
                    existing_dict[key] = {}
                existing_dict[key] = self.append_or_create_metric_lists(
                    existing_dict[key], new_value)
            else:
                if key not in existing_dict:
                    existing_dict[key] = []
                existing_dict[key].append(new_value)
        
        return existing_dict

    def get_best_parameters(self, param_type, server_type):
        """Get best hyperparameter value for given server type."""
        results = self.load_results(param_type)
        if results is None:
            return None
    
        
        # Collect metrics across all learning rates for this server
        server_metrics = {}
        for lr in results.keys():
            if server_type not in results[lr]:
                continue
            server_metrics[lr] = results[lr][server_type]
        
        if not server_metrics:
            return None

        return self._select_best_hyperparameter(server_metrics)

    def _select_best_hyperparameter(self, lr_results):
        """
        Select best hyperparameter based on minimum median loss across all rounds.
        """
        best_loss = float('inf')
        best_param = None
        
        for lr, metrics in lr_results.items():
            # For each round, calculate mean loss across all runs
            num_rounds = len(metrics['global']['losses'][0])  # Assuming all runs have same number of rounds
            
            # Calculate median loss for each round
            round_mean_losses = []
            for round_idx in range(num_rounds):
                # Get losses for this round across all runs
                round_losses = [run[round_idx] for run in metrics['global']['losses']]
                median_loss = np.mean(round_losses)
                round_mean_losses.append(median_loss)
            
            # Find the best (lowest) mean loss across all rounds
            best_round_mean_loss = min(round_mean_losses)
            
            # Update best parameter if this learning rate achieved a better loss
            if best_round_mean_loss < best_loss:
                best_loss = best_round_mean_loss
                best_param = lr
        
        return best_param