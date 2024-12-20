from configs import *
from helper import *

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

    best_metrics = {
            'loss': float('inf'),
            'accuracy': 0,
            'balanced_accuracy': 0,
            'f1_macro': 0,
            'f1_weighted': 0,
            'mcc': 0
        }

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
    
    def update_best_metrics(self, metrics_dict, loss):
        """Update best metrics if improved."""
        updated = False
        
        # Check loss improvement
        if loss < self.best_metrics['loss']:
            self.best_metrics['loss'] = loss
            updated = True
            
        # Check each metric improvement
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = value
                updated = True
            elif value > self.best_metrics[metric_name]: 
                self.best_metrics[metric_name] = value
                updated = True
                
        return updated

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
        return {
            'accuracy': (predictions == labels).mean(),
            'balanced_accuracy': balanced_accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_weighted': f1_score(labels, predictions, average='weighted'),
            'mcc': matthews_corrcoef(labels, predictions)
        }


class Client:
    """Client class that handles both model management and training."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 modelstate: ModelState,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False):
        self.config = config
        self.data = data
        self.device = config.device
        self.metrics_calculator = metrics_calculator

        # Initialize model states
        self.global_state = modelstate

        # Create personal state if needed
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
            metrics = self.metrics_calculator.calculate_metrics(
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_param = self.config.personalization_params['reg_param']

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_param = self.config.personalization_params['reg_param']

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_param = self.config.personalization_params['reg_param']

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LayerClient(Client):
    """Client for layer-wise federated learning."""
    def __init__(self, *args, **kwargs):
        self.layers_to_include = kwargs['config'].personalization_params['layers_to_include']
        super().__init__(*args, **kwargs)

    def set_model_state(self, state_dict, personal = False):
        """Selectively load state dict only for federated layers."""
        state = self.personal_state if personal else self.global_state
        selective_load_state_dict(state.model, state_dict, self.layers_to_include)

class BABUClient(LayerClient):
    """Client implementation for BABU."""
    def set_head_body_training(self, train_head):
        model = self.global_state.model
        head_params = []
        body_params = []
        
        for name, param in model.named_parameters():
            is_head = not any(layer_name in name for layer_name in self.layers_to_include)
            param.requires_grad = train_head if is_head else not train_head
            
            # Track parameters for logging
            if is_head:
                head_params.append(name)
            else:
                body_params.append(name)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize with body training
        self.set_head_body_training(train_head=False)

    def train_head(self):
        """Train only the head of the model."""
        self.set_head_body_training(train_head=True)
        return self.train(personal=False)

class FedLPClient(Client):
    """Client for FedLP with layer-wise probabilistic participation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get layer-preserving rates (LPRs) for each layer
        self.layer_preserving_rate = self.config.personalization_params['layer_preserving_rate']
        
    def generate_layer_indicators(self):
        """Generate binary indicators for all model layers using a single preservation rate."""
        indicators = {}
        for name, _ in self.global_state.model.named_parameters():
            # Get base layer name (e.g., 'conv1' from 'conv1.weight')
            layer_name = name.split('.')[0]
            if layer_name not in indicators:  # Only generate once per layer
                # Single Bernoulli trial with same probability for each layer
                indicators[layer_name] = 1 if random.random() < self.layer_preserving_rate else 0
        return indicators
        
    def get_pruned_model_state(self):
        """Get model state with pruned layers based on indicators."""
        indicators = self.generate_layer_indicators()
        state_dict = self.global_state.model.state_dict()
        pruned_state = {}
        
        for name, param in state_dict.items():
            # Get layer name from parameter name
            layer_name = name.split('.')[0]
            if layer_name in indicators and indicators[layer_name]:
                pruned_state[name] = param
                
        return pruned_state, indicators
    

class FedLAMAClient(Client):
    """Client for FedLAMA."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class pFedLAClient(Client):
    """pFedLA client implementation using hypernetwork for layer-wise aggregation."""
    def __init__(self, *args, **kwargs):
        # Initialize with personal_model=False since pFedLA handles personalization differently
        super().__init__(*args, **kwargs)
        
        # Store initial parameters for computing updates
        self.initial_params = {}
        self._store_initial_params()

    def _store_initial_params(self):
        """Store initial model parameters."""
        for name, param in self.global_state.model.state_dict().items():
            self.initial_params[name] = param.clone().detach()

    def set_model_state(self, state_dict, personal=False):
        """Override to update initial params when model state is set."""
        super().set_model_state(state_dict, personal=False) 
        self._store_initial_params()

    def train(self, personal=False):
        """Train model and return parameter updates."""
        # Train for specified number of epochs
        final_loss = super().train(personal=False)
        
        # Compute updates
        updates = self.compute_updates()
        
        # Store current state as initial state for next round
        self._store_initial_params()
        
        return updates, {
            'loss': final_loss,
            'model_state': self.global_state.model.state_dict()
        }

    def compute_updates(self):
        """Compute parameter updates from initial state."""
        updates = OrderedDict()
        current_params = self.global_state.model.state_dict()
        
        for name, current_param in current_params.items():
            if name in self.initial_params:
                updates[name] = current_param - self.initial_params[name]
            else:
                updates[name] = torch.zeros_like(current_param)
                
        return updates
