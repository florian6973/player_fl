"""
Module for calculating detailed model analysis metrics including layer weights,
gradient importance, Hessian properties, and activation similarity.
"""
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from helper import move_to_device, cleanup_gpu 
from configs import *

cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 6))


class ModelDeviceManager:
    """Context manager to temporarily move a model to a device."""
    def __init__(self, model: nn.Module, device: str, eval_mode: bool = False):
        self.model = model
        self.device = device
        self.original_device = next(model.parameters()).device
        self.original_mode = model.training
        self.eval_mode = eval_mode

    def __enter__(self):
        self.model.to(self.device)
        if self.eval_mode:
            self.model.eval()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.to(self.original_device)
        if self.eval_mode and self.original_mode: # Restore original mode if it was training
            self.model.train()
        cleanup_gpu()


def _model_weight_data(param: Tensor) -> Dict[str, float]:
    """ Get data on the weights per layer including mean, variance and % above certain magnitude"""
    param_cpu = param.detach().cpu()
    data = {
        'Weight Mean': abs(param_cpu).mean().item(),
        'Weight Variance': param_cpu.var().item(),
    }
    for threshold in [0.01, 0.05, 0.1]:
        small_weights = (param_cpu.abs() < threshold).sum().item()
        data[f'% Weights < {threshold}'] = 100 * small_weights / param_cpu.numel() if param_cpu.numel() > 0 else 0
    return data

def _model_weight_importance(param: Tensor, gradient: Tensor) -> Dict[str, float]:
    """ Estimate model layer importance using first order gradient approx """
    param_cpu = param.detach().cpu()
    grad_cpu = gradient.detach().cpu()
    importance = (param_cpu * grad_cpu).pow(2).sum().item()
    importance_per = importance / param_cpu.numel() if param_cpu.numel() > 0 else 0
    grad_var = grad_cpu.var().item()
    return {
        'Gradient Importance': importance,
        'Gradient Importance per': importance_per,
        'Gradient Variance': grad_var
    }

# Simplified Hessian computation for integration
# NOTE: Full Hessian computation is very memory intensive.
# This version calculates eigenvalues using SVD on smaller layers or estimates.
def _compute_eigenvalues(matrix: Tensor) -> Tuple[np.ndarray, float]:
    """Compute SVD eigenvalues (singular values) for a matrix."""
    try:
        # Using SVD as a proxy for Hessian eigenvalues, common in optimization literature
        # Ensure matrix is float32 for SVD
        eigenvalues = torch.linalg.svdvals(matrix.float())
        sum_eigenvalues = torch.sum(eigenvalues).item()
        return eigenvalues.cpu().numpy(), sum_eigenvalues
    except torch.linalg.LinAlgError:
        print(f"Warning: SVD computation failed for matrix of shape {matrix.shape}. Returning zeros.")
        num_eigenvalues = min(matrix.shape)
        return np.zeros(num_eigenvalues), 0.0
    except Exception as e:
        print(f"Warning: Error during SVD computation for matrix of shape {matrix.shape}: {e}. Returning zeros.")
        num_eigenvalues = min(matrix.shape)
        return np.zeros(num_eigenvalues), 0.0


def _hessian_metrics(param: Tensor, loss: Tensor, calculate_hessian: bool) -> Dict[str, Union[float, np.ndarray]]:
    """ Calculate Hessian-related metrics for a layer."""
    metrics = {
        'SVD Sum EV': np.nan,
        'EV Skewness': np.nan,
        '% EV small': np.nan,
        '% EV neg': np.nan, # Note: SVD values are non-negative
        'Gradient Importance 2': np.nan,
        'Gradient Importance per 2': np.nan,
        'Hessian Variance': np.nan, # Placeholder, actual Hessian variance is hard
        'Condition Number': np.nan,
        'Operator norm': np.nan,
        'SVD Eigenvalues': np.array([]), # Store the actual values if needed
    }
    if not calculate_hessian or not param.requires_grad or param.grad is None or loss is None:
        return metrics

    try:
        # Compute gradient first
        first_grads = torch.autograd.grad(loss, param, create_graph=True, allow_unused=True)[0]
        if first_grads is None:
            print(f"Warning: Could not compute gradient for Hessian calculation for parameter shape {param.shape}.")
            return metrics

        # Hessian-vector product (approximation) - less memory than full Hessian
        # Use a random vector (Rademacher or Gaussian)
        v = torch.randn_like(param)
        v = v / torch.norm(v) # Normalize

        Hv = torch.autograd.grad(first_grads, param, grad_outputs=v, retain_graph=False, allow_unused=True)[0]

        if Hv is None:
            print(f"Warning: Could not compute Hessian-vector product for parameter shape {param.shape}.")
            return metrics

        Hv_cpu = Hv.detach().cpu()
        param_cpu = param.detach().cpu()

        # Second-order importance estimate
        importance2 = (param_cpu * Hv_cpu).pow(2).sum().item()
        importance_per2 = importance2 / param_cpu.numel() if param_cpu.numel() > 0 else 0

        metrics['Gradient Importance 2'] = importance2
        metrics['Gradient Importance per 2'] = importance_per2
        metrics['Hessian Variance'] = Hv_cpu.var().item() # Variance of H*v

        # Compute eigenvalues only for manageable layers (e.g., linear layers)
        # Full Hessian computation and eigenvalue decomposition is often too expensive.
        # We use SVD values as an approximation here.
        if param.dim() == 2: # Potentially a linear layer
            matrix_for_svd = param.detach().float() # Or first_grads if analyzing gradient geometry
            svd_e, svd_sum_e = _compute_eigenvalues(matrix_for_svd)

            if len(svd_e) > 0:
                metrics['SVD Eigenvalues'] = svd_e
                metrics['SVD Sum EV'] = svd_sum_e
                metrics['Operator norm'] = svd_e.max().item()
                min_positive_ev = svd_e[svd_e > 1e-8].min() if np.any(svd_e > 1e-8) else 1e-8
                metrics['Condition Number'] = metrics['Operator norm'] / min_positive_ev
                metrics['EV Skewness'] = scipy.stats.skew(svd_e)
                threshold = 0.01 * metrics['Operator norm']
                metrics['% EV small'] = 100 * np.sum(svd_e < threshold) / len(svd_e) if len(svd_e) > 0 else 0
                # SVD values are non-negative
                metrics['% EV neg'] = 0.0

    except RuntimeError as e:
        print(f"Warning: Hessian calculation failed for param shape {param.shape}: {e}")
        # Reset metrics if calculation failed mid-way
        metrics = {k: np.nan if isinstance(v, float) else (np.array([]) if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    finally:
        # Ensure gradients from this calculation don't affect subsequent steps
        if param.grad is not None:
            param.grad = None # Or param.grad.zero_() if preferred

    return metrics

def _analyse_layer(name: str, param: Tensor, loss: Optional[Tensor], calculate_hessian: bool) -> Optional[Dict[str, float]]:
    """Analyse a single layer's weights and gradients."""
    if param is None or param.grad is None:
        # print(f"Skipping layer {name}: No parameter or gradient.")
        return None

    layer_data = _model_weight_data(param)
    layer_data.update(_model_weight_importance(param, param.grad))

    hessian_data = _hessian_metrics(param, loss, calculate_hessian)
    layer_data.update(hessian_data)

    return layer_data

def calculate_local_layer_metrics(model: nn.Module,
                                  data_batch: Union[Tensor, Tuple],
                                  criterion: nn.Module,
                                  device: str,
                                  calculate_hessian: bool = False) -> pd.DataFrame:
    """
    Calculates weight stats, gradient importance, and optionally Hessian metrics
    for each layer of the model using a single data batch.
    """
    layer_data_dict = {}
    loss_value = None # Initialize loss_value

    try:
        with ModelDeviceManager(model, device) as model_on_device:
            model_on_device.train() # Ensure model is in train mode for gradients
            features, labels = data_batch
            features = move_to_device(features, device)
            labels = move_to_device(labels, device)

            model_on_device.zero_grad()
            outputs = model_on_device(features)
            loss = criterion(outputs, labels)
            loss_value = loss.detach() # Store loss for Hessian calculation if needed

            # Backward pass: Create graph only if Hessian needed
            loss.backward(create_graph=calculate_hessian, retain_graph=calculate_hessian)

            # --- Analysis ---
            named_params = {name: param for name, param in model_on_device.named_parameters() if param.requires_grad}

            for name, param in named_params.items():
                # Skip biases and layers explicitly excluded if needed
                if "bias" in name: # Simple exclusion example
                    continue
                if param.grad is None:
                    # This might happen if a layer isn't used in the forward pass for this batch
                    # or due to retain_graph issues if Hessian calculation failed partially.
                    print(f"Warning: Gradient is None for layer {name}. Skipping analysis.")
                    continue

                # Process full layers (FC, Embeddings) or per-channel (Conv)
                if param.dim() == 4:  # Basic check for Conv layer (e.g., Conv2d weight [out_c, in_c, kH, kW])
                    out_channels = param.size(0)
                    base_layer_name = name.split('.')[0] # Group channels under layer name
                    # Analyse channel-wise (as in old code)
                    # Note: Hessian per channel can be very slow. Consider layer-level approx if needed.
                    for channel in range(out_channels):
                        param_channel = param[channel]
                        grad_channel = param.grad[channel]

                        # Re-calculating loss/grads per channel isn't feasible.
                        # Use the overall loss and channel grads for analysis.
                        # Note: Hessian metrics here would be based on the overall loss graph.
                        # Recreating per-channel Hessians is complex. Using overall Hessian approximation.
                        channel_metrics = _analyse_layer(f"{base_layer_name}_c_{channel}", param_channel, loss_value, calculate_hessian=False) # Disable channel-Hessian by default
                        if channel_metrics:
                            layer_data_dict[f"{base_layer_name}_c_{channel}"] = channel_metrics

                elif param.dim() <= 2 : # FC layers, embeddings etc.
                    layer_metrics = _analyse_layer(name, param, loss_value, calculate_hessian)
                    if layer_metrics:
                        # Use a more general layer name if possible
                        layer_name_parts = name.split('.')[:-1] # Remove '.weight' or '.bias'
                        simple_name = '.'.join(layer_name_parts) if layer_name_parts else name
                        layer_data_dict[simple_name] = layer_metrics
                else:
                     print(f"Skipping layer {name} with unexpected dimension {param.dim()}")


            # Explicitly zero gradients after analysis
            model_on_device.zero_grad(set_to_none=True)

    except Exception as e:
        print(f"Error during local layer metrics calculation: {e}")
        raise # Re-raise the exception after printing
    finally:
        # Ensure cleanup happens
        cleanup_gpu()


    if not layer_data_dict:
        print("Warning: No layer metrics were calculated.")
        return pd.DataFrame()

    # Filter out layers where analysis might have failed (all NaN)
    valid_layer_data = {name: metrics for name, metrics in layer_data_dict.items() if metrics and not all(np.isnan(v) for v in metrics.values() if isinstance(v, (float, np.floating))) }

    return pd.DataFrame.from_dict(valid_layer_data, orient='index')


# --- Activation Similarity Logic ---

_activation_hooks = []
_activation_dict: Dict[str, List[Tuple[str, np.ndarray]]] = {}

def _hook_fn(layer_name: str, site_id: str):
    """Hook function to capture activations."""
    def hook(module, input, output):
        global _activation_dict
        # Detach, move to CPU, convert to NumPy
        activation = output.detach().cpu().numpy()
        if site_id not in _activation_dict:
            _activation_dict[site_id] = []
        _activation_dict[site_id].append((layer_name, activation))
    return hook

def _register_hooks(model: nn.Module, site_id: str):
    """Registers forward hooks on relevant modules."""
    global _activation_hooks
    # Clear previous hooks if any
    for handle in _activation_hooks:
        handle.remove()
    _activation_hooks = []

    for name, module in model.named_modules():
        # Example: Hook convolutional and linear layers, adjust as needed
        # Avoid hooking container modules like Sequential or the top-level model itself
        is_leaf_module = '.' not in name and not list(module.children())
        is_relevant_type = isinstance(module, (nn.Conv2d, nn.Linear, nn.TransformerEncoderLayer, nn.Embedding)) # Add other relevant types

        # Hooking modules based on type might be more robust than name checks like '.'
        if is_relevant_type:
           # Use module name directly if available, otherwise generate one
           hook_name = name if name else f"type_{type(module).__name__}"
           handle = module.register_forward_hook(_hook_fn(hook_name, site_id))
           _activation_hooks.append(handle)

def _remove_hooks():
    """Removes all registered hooks."""
    global _activation_hooks
    for handle in _activation_hooks:
        handle.remove()
    _activation_hooks = []


def get_model_activations(model: nn.Module,
                          probe_data_batch: Union[Tensor, Tuple],
                          device: str,
                          site_id: str) -> List[Tuple[str, np.ndarray]]:
    """
    Runs a probe data batch through the model and captures activations
    from registered hooks.
    """
    global _activation_dict
    _activation_dict[site_id] = [] # Reset activations for this site_id

    try:
        with ModelDeviceManager(model, device, eval_mode=True) as model_on_device:
            _register_hooks(model_on_device, site_id)

            features, _ = probe_data_batch # Labels usually not needed for activation capture
            features = move_to_device(features, device)

            with torch.no_grad():
                _ = model_on_device(features)

            _remove_hooks() # Important to remove hooks after use

    except Exception as e:
        print(f"Error during activation extraction for site {site_id}: {e}")
        _remove_hooks() # Ensure hooks are removed even on error
        raise
    finally:
        cleanup_gpu()

    return _activation_dict.get(site_id, [])


def calculate_activation_similarity(activations_dict: Dict[str, List[Tuple[str, np.ndarray]]],
                                    probe_data_batch: Optional[Union[Tensor, Tuple]] = None,
                                    num_sites: int = 0,
                                    cpus: int = 4) -> Dict[str, pd.DataFrame]:
    """
    Calculates pairwise similarity between site activations layer by layer using NetRep.
    """
    if not activations_dict:
        return {}

    comparison_data = {}
    site_ids = list(activations_dict.keys())
    if num_sites == 0:
        num_sites = len(site_ids)

    # Assume all sites have the same layers in the same order
    # Get layer names from the first site
    if not site_ids: return {}
    first_site_id = site_ids[0]
    if not activations_dict[first_site_id]: return {}

    layer_names = [name for name, _ in activations_dict[first_site_id]]

    # Extract mask if attention model data is provided
    mask = None
    is_attention_model = False
    if probe_data_batch and isinstance(probe_data_batch[0], (tuple, list)):
         _, potential_mask = probe_data_batch[0]
         if isinstance(potential_mask, Tensor) and potential_mask.dtype == torch.bool:
             mask = potential_mask.cpu().numpy()
             is_attention_model = True
         elif isinstance(potential_mask, Tensor) and potential_mask.dtype in [torch.float, torch.long]: # Handle float/int masks
             mask = potential_mask.cpu().numpy().astype(bool)
             is_attention_model = True


    for layer_idx, layer_name in enumerate(layer_names):
        print(f"Comparing activations for layer: {layer_name}")
        # Gather activations for this layer from all sites that have it
        layer_activations = []
        valid_site_ids_for_layer = []
        for site_id in site_ids:
             if layer_idx < len(activations_dict[site_id]) and activations_dict[site_id][layer_idx][0] == layer_name:
                 layer_activations.append(activations_dict[site_id][layer_idx][1])
                 valid_site_ids_for_layer.append(site_id)
             else:
                 print(f"Warning: Layer mismatch or missing layer {layer_name} for site {site_id}")

        if len(layer_activations) < 2:
            print(f"Skipping layer {layer_name}: Need at least two sites with activations.")
            continue

        current_num_sites = len(valid_site_ids_for_layer)
        act_shape = layer_activations[0].shape
        is_conv_layer = len(act_shape) == 4  # NCHW format
        is_transformer_layer = (len(act_shape) == 3 or 'embedding' in layer_name.lower()) and is_attention_model # N, SeqLen, Dim

        metric = LinearMetric(alpha=1.0, center_columns=True, score_method="angular") # As in old code

        try:
            result_matrix = np.zeros((current_num_sites, current_num_sites))

            if is_conv_layer:
                # Transpose to NHWC format for convolve_metric
                acts_nhwc = [np.transpose(act, (0, 2, 3, 1)) for act in layer_activations]
                # Calculate pairwise distances using convolve_metric
                # Note: convolve_metric calculates a single distance value. We need pairwise.
                # Replicating the pairwise loop from old code:
                for i, j in combinations(range(current_num_sites), 2):
                     dist = convolve_metric(metric, acts_nhwc[i], acts_nhwc[j], processes=cpus)
                     min_dist = dist.min() # Assuming this is the desired metric from old code
                     result_matrix[i, j] = min_dist
                     result_matrix[j, i] = min_dist
                del acts_nhwc

            elif is_transformer_layer and mask is not None:
                 # Apply masking: Mask shape needs to align with activation shape (N, SeqLen, Dim)
                 # Assume mask is (N, SeqLen)
                 mask_expanded = np.expand_dims(mask[:act_shape[0], :act_shape[1]], axis=-1).astype(float) # Match batch and seq len, add dim
                 masked_acts = [act * mask_expanded for act in layer_activations]
                 # Sum over sequence length dimension (dim=1)
                 pooled_acts = [np.sum(masked_act, axis=1) for masked_act in masked_acts] # Result shape (N, Dim)
                 # Use standard pairwise distances on pooled representations
                 _, result_matrix = metric.pairwise_distances(pooled_acts, pooled_acts, processes=cpus)
                 del masked_acts, pooled_acts

            else: # Default for FC layers or others
                # Flatten if necessary? Assuming shape is (N, Features) or similar
                acts_flat = [act.reshape(act.shape[0], -1) for act in layer_activations]
                _, result_matrix = metric.pairwise_distances(acts_flat, acts_flat, processes=cpus)
                del acts_flat

            # Create DataFrame with proper site_id indexing
            result_df = pd.DataFrame(result_matrix, index=valid_site_ids_for_layer, columns=valid_site_ids_for_layer)
            comparison_data[layer_name] = result_df

        except Exception as e:
            print(f"Error calculating similarity for layer {layer_name}: {e}")
            # Optionally store NaN or skip the layer
            comparison_data[layer_name] = pd.DataFrame(np.nan, index=valid_site_ids_for_layer, columns=valid_site_ids_for_layer)

        finally:
            # Clear intermediate activations for the layer to save memory
            del layer_activations
            gc.collect()


    return comparison_data