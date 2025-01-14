ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DATA_DIR = f'{ROOT_DIR}/data'
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
    
    if server_type in ['layerpfl', 'layerpfl_minus_1', 'layerpfl_plus_1']:
        params['layers_to_include'] = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]
        params['reg_param'] = REG_PARAMS['layerpfl'][dataset_name]    # Optional, not currently used
    
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

def move_model_to_device(model, device):
    """Safely move model to device with proper cleanup."""
    try:
        return model.to(device)
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()

def cleanup_gpu():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def move_to_device(batch, device):
    """Move batch data to device, handling both single tensors and lists/tuples of tensors."""
    if isinstance(batch, (list, tuple)):
        return [x.to(device) if x is not None else None for x in batch]
    return batch.to(device)