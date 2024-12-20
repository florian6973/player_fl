ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DATA_DIR = f'{ROOT_DIR}/data'
import torch
import pandas as pd
import numpy as np


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


def compute_proximal_term(model_params, reference_params, reg_param):
    """Calculate proximal term between two sets of model parameters."""
    proximal_term = 0.0
    for param, ref_param in zip(model_params, reference_params):
        proximal_term += (reg_param / 2) * torch.norm(param - ref_param) ** 2
    return proximal_term

def add_gradient_regularization(model_params, reference_params, reg_param):
    """Add regularization directly to gradients."""
    for param, ref_param in zip(model_params, reference_params):
        if param.grad is not None:
            reg_term = reg_param * (param - ref_param)
            param.grad.add_(reg_term)


def get_layer_params(model, layers_to_include, for_federated = True):
    """Get parameters split by layer inclusion."""
    included_params = []
    excluded_params = []
    
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_include):
            included_params.append(param)
        else:
            excluded_params.append(param)
            
    return included_params if for_federated else excluded_params

def selective_load_state_dict(model, state_dict, layers_to_include):
    """Load state dict only for specified layers."""
    current_state = model.state_dict()
    for name, param in state_dict.items():
        if any(layer in name for layer in layers_to_include):
            current_state[name].copy_(param)
    model.load_state_dict(current_state)