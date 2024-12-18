ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
import copy
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/run_models')
from sklearn import metrics
from torch.utils.data  import DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from helper import *
from itertools import combinations
from functools import partial
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score

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


@dataclass
class SiteData:
    """Holds DataLoader and metadata for a site."""
    site_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    weight: float = 1.0
    
    def __post_init__(self):
        if self.train_loader is not None:
            self.num_samples = len(self.train_loader.dataset)

@dataclass
class ModelState:
    """Holds state for a single model (global or personalized)."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    best_loss: float = float('inf')
    best_model: Optional[nn.Module] = None
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.best_model is None and self.model is not None:
            self.best_model = copy.deepcopy(self.model)
    
    def copy(self):
        """Create a new ModelState with copied model and optimizer."""
        # Create new model instance
        new_model = copy.deepcopy(self.model)
        
        # Setup optimizer
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)
        new_optimizer.load_state_dict(optimizer_state)
        
        # Create new model state
        return ModelState(
            model=new_model,
            optimizer=new_optimizer,
            criterion= self.criterion 
        )

class MetricsCalculator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def process_predictions(self, predictions, labels):
        """Process model predictions"""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = predictions.argmax(axis=1)
            
        return predictions, labels

    def calculate_metrics(self, predictions, labels):
        """Calculate multiple classification metrics."""
        y_pred, y_true = self.process_predictions(self, predictions, labels)
        return {
            'accuracy': (y_pred == y_true).mean(),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }


class Client:
    """Client class that handles both model management and training."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False):
        self.config = config
        self.data = data
        self.device = config.device
        self.metrics_calculator = metrics_calculator

        # Initialize model states
        self.global_state = ModelState(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        self.personal_state = self.global_state.copy() if personal_model else None

    def get_model_state(self, personal = False):
        """Get model state dictionary."""
        state = self.personal_state if personal else self.global_state
        return state.model.state_dict()

    def set_model_state(self, state_dict, personal = False):
        """Set model state from dictionary."""
        state = self.personal_state if personal else self.global_state
        state.model.load_state_dict(state_dict)

    def update_best_model(self, loss, personal  = False):
        """Update best model if loss improves."""
        state = self.personal_state if personal else self.global_state
        if loss < state.best_loss:
            state.best_loss = loss
            state.best_model = copy.deepcopy(state.model)
            return True
        return False

    def train_epoch(self, personal = False):
        """Train for one epoch."""
        state = self.personal_state if personal else self.global_state
        model = state.model.train().to(self.device)
        total_loss = 0.0
        
        try:
            for batch_x, batch_y in self.data.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                loss.backward()
                
                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.clip_value
                    )
                
                state.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data.train_loader)
            state.train_losses.append(avg_loss)

                
            return avg_loss
            
        finally:
            model.to('cpu')
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def train(self, personal = False):
        """Train for multiple epochs."""
        final_loss = 0.0
        for epoch in range(self.config.epochs):
            final_loss = self.train_epoch(personal)
        return final_loss

    def evaluate(self, loader, personal = False, validate = False):
        """Evaluate model performance."""
        state = self.personal_state if personal else self.global_state
        model = (state.model if validate else state.best_model).to(self.device)
        model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        try:
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = state.criterion(outputs, batch_y)
                    total_loss += loss.item()
                    
                    predictions, labels = self.metrics_calculator.process_predictions(
                        outputs, batch_y
                    )
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)

            avg_loss = total_loss / len(loader)
            metrics = self.metrics_calculator.calculate_score(
                np.array(all_labels),
                np.array(all_predictions)
            )
            
            return avg_loss, metrics
            
        finally:
            model.to('cpu')
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def validate(self, personal = False):
        """Validate current model."""
        state = self.personal_state if personal else self.global_state
        val_loss, val_metrics = self.evaluate(
            self.data.val_loader, 
            personal, 
            validate=True
        )
        
        state.val_losses.append(val_loss)
        state.val_scores.append(val_metrics)
        
        self.update_best_model(val_loss, personal)
        return val_loss, val_metrics

    def test(self, personal = False):
        """Test using best model."""
        state = self.personal_state if personal else self.global_state
        test_loss, test_metrics = self.evaluate(
            self.data.test_loader,
            personal
        )
        
        state.test_losses.append(test_loss)
        state.test_scores.append(test_metrics)
        
        return test_loss, test_metrics

class FedProxClient(Client):
    """FedProx client implementation."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = True):
        super().__init__(
            config=config,
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics_calculator=metrics_calculator,
            personal_model=personal_model
        )
        self.reg_param = config.personalization_params['reg_param']
    def train_epoch(self, personal=False):
        state = self.global_state  # FedProx only uses global model
        model = move_model_to_device(state.model.train(), self.device)
        total_loss = 0.0
        
        try:
            for batch_x, batch_y in self.data.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                
                proximal_term = compute_proximal_term(
                    model.parameters(),
                    self.global_state.model.parameters(),
                    self.reg_param
                )
                
                total_loss_batch = loss + proximal_term
                total_loss_batch.backward()
                
                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.clip_value
                    )
                
                state.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data.train_loader)
            state.train_losses.append(avg_loss)
            return avg_loss
            
        finally:
            model.to('cpu')
            cleanup_gpu()

class PFedMeClient(Client):
    """PFedMe client implementation with proximal regularization."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = True):
        super().__init__(
            config=config,
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics_calculator=metrics_calculator,
            personal_model=personal_model
        )
        self.reg_param = config.personalization_params['reg_param']

    def train_epoch(self, personal=True):
        """Train for one epoch with proximal term regularization."""
        # PFedMe primarily uses personal model
        state = self.personal_state if personal else self.global_state
        model = move_model_to_device(state.model.train(), self.device)
        global_model = move_model_to_device(self.global_state.model, self.device)
        total_loss = 0.0
        
        try:
            for batch_x, batch_y in self.data.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                
                proximal_term = compute_proximal_term(
                    model.parameters(),
                    global_model.parameters(),
                    self.reg_param
                )
                
                total_batch_loss = loss + proximal_term
                total_batch_loss.backward()
                
                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.clip_value
                    )
                
                state.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data.train_loader)
            state.train_losses.append(avg_loss)
            return avg_loss
            
        finally:
            model.to('cpu')
            global_model.to('cpu')
            cleanup_gpu()

    def train(self, personal=True):
        """Main training loop defaulting to personal model."""
        final_loss = 0.0
        for epoch in range(self.config.epochs):
            final_loss = self.train_epoch(personal)
        return final_loss

class DittoClient(Client):
    """Ditto client implementation."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = True):
        super().__init__(
            config=config,
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics_calculator=metrics_calculator,
            personal_model=personal_model
        )
        self.reg_param = config.personalization_params['reg_param']

    def train_epoch(self, personal=False):
        if not personal:
            return super().train_epoch(personal=False)
            
        state = self.personal_state
        model = move_model_to_device(state.model.train(), self.device)
        global_model = move_model_to_device(self.global_state.model, self.device)
        total_loss = 0.0
        
        try:
            for batch_x, batch_y in self.data.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                loss.backward()
                
                add_gradient_regularization(
                    model.parameters(),
                    global_model.parameters(),
                    self.reg_param
                )
                
                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.clip_value
                    )
                    
                state.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data.train_loader)
            state.train_losses.append(avg_loss)
            return avg_loss
            
        finally:
            model.to('cpu')
            global_model.to('cpu')
            cleanup_gpu()

class LocalAdaptationClient(Client):
    """Client that performs additional local training after federation."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False):
        super().__init__(
            config=config,
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics_calculator=metrics_calculator,
            personal_model=personal_model
        )


class Server:
    """Base server class for federated learning."""
    def __init__(self, config: TrainerConfig, model: nn.Module):
        self.config = config
        self.device = config.device
        self.clients = {}
        self.global_model = copy.deepcopy(model)
        self.best_model = copy.deepcopy(model)
        self.best_loss = float('inf')
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.test_losses = []
        self.test_scores = []

    def add_client(self, client_id, client: Client):
        """Add a client to the federation."""
        self.clients[client_id] = client
        self._update_client_weights()

    def _update_client_weights(self):
        """Update client weights based on dataset sizes."""
        total_samples = sum(client.data.num_samples for client in self.clients.values())
        for client in self.clients.values():
            client.data.weight = client.data.num_samples / total_samples

    def train_round(self, personal = False):
        """Run one round of training."""
        # Train all clients
        train_loss = 0
        val_loss = 0
        val_score = 0

        for client in self.clients.values():
            # Train and validate
            client_train_loss = client.train(personal)
            client_val_loss, client_val_score = client.validate(personal)
            
            # Weight metrics by client dataset size
            train_loss += client_train_loss * client.data.weight
            val_loss += client_val_loss * client.data.weight
            val_score += client_val_score * client.data.weight

        # Track metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_scores.append(val_score)

        # Aggregate and distribute
        self.aggregate_models(personal)
        self.distribute_global_model()

        # Update best model if improved
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(self.global_model)

        return train_loss, val_loss, val_score

    def test_global(self, personal = False):
        """Test the global model across all clients."""
        test_loss = 0
        test_score = 0

        for client in self.clients.values():
            client_loss, client_score = client.test(personal)
            test_loss += client_loss * client.data.weight
            test_score += client_score * client.data.weight

        self.test_losses.append(test_loss)
        self.test_scores.append(test_score)

        return test_loss, test_score

class FLServer(Server):
    """Base federated learning server with FedAvg implementation."""
    def aggregate_models(self, personal = False):
        """Standard FedAvg aggregation."""
        # Reset global model parameters
        for param in self.global_model.parameters():
            param.data.zero_()
            
        # Aggregate parameters
        for client in self.clients.values():
            client_model = client.personal_state.model if personal else client.global_state.model
            for g_param, c_param in zip(self.global_model.parameters(), client_model.parameters()):
                g_param.data.add_(c_param.data * client.data.weight)

    def distribute_global_model(self):
        """Distribute global model to all clients."""
        global_state = self.global_model.state_dict()
        for client in self.clients.values():
            client.set_model_state(global_state)

class FedAvgServer(FLServer):
    """Standard FedAvg implementation."""
    pass

class FedProxServer(FLServer):
    """FedProx server implementation."""
    pass  # Uses standard FedAvg behavior

class PFedMeServer(FLServer):
    """PFedMe server implementation."""
    def train_round(self, personal = True):
        return super().train_round(personal)
    
    def test_global(self, personal = True):
        return super().test_global(personal)

class DittoServer(FLServer):
    """Ditto server implementation."""
    def train_round(self, personal = True):
        return super().train_round(personal)
    
    def test_global(self, personal = True):
        return super().test_global(personal)

class LocalAdaptationServer(FLServer):
    """Local adaptation server implementation."""
    def train_round(self, personal = False, final_round = False):
        """Run one round of training with optional final round behavior."""
        # Use parent class training logic
        train_loss, val_loss, val_score = super().train_round(personal)

        if final_round:
            train_loss = 0
            val_loss = 0
            val_score = 0
            # Restore best models to clients for final evaluation
            for client in self.clients.values():
                client_train_loss = client.train(personal)
                client_val_loss, client_val_score = client.validate(personal)
            
            # Weight metrics by client dataset size
            train_loss += client_train_loss * client.data.weight
            val_loss += client_val_loss * client.data.weight
            val_score += client_val_score * client.data.weight


        return train_loss, val_loss, val_score


def compute_proximal_term(model_params, reference_params, mu: float) -> torch.Tensor:
    """Calculate proximal term between two sets of model parameters."""
    proximal_term = 0.0
    for param, ref_param in zip(model_params, reference_params):
        proximal_term += (mu / 2) * torch.norm(param - ref_param) ** 2
    return proximal_term

def add_gradient_regularization(model_params, reference_params, reg_param: float):
    """Add regularization directly to gradients."""
    for param, ref_param in zip(model_params, reference_params):
        if param.grad is not None:
            reg_term = reg_param * (param - ref_param)
            param.grad.add_(reg_term)

def move_model_to_device(model: nn.Module, device: str) -> nn.Module:
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