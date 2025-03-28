ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from helper import move_to_device, cleanup_gpu 
from configs import  *
from clients import Client
from layer_analytics import  *


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
    requires_personal_model: bool = False
    algorithm_params: Optional[Dict] = None
    num_cpus: int = 4 # Added for analysis

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
            self.best_model = copy.deepcopy(self.model).to(next(self.model.parameters()).device)
    
    def copy(self):
        """Create a new ModelState with copied model and optimizer."""
        # Create new model instance
        new_model = copy.deepcopy(self.model).to(next(self.model.parameters()).device)
        
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

class AnalyticsClient(Client):
    """
    Extends the base Client to add methods for local model analysis and
    activation extraction.
    """
    def __init__(self,
                 config: TrainerConfig,
                 data: SiteData,
                 modelstate: ModelState,
                 metrics_calculator: None,
                 personal_model: bool = False):
        super().__init__(config, data, modelstate, metrics_calculator, personal_model)
        print(f"AnalyticsClient created for site {self.data.site_id}")

    def train_round(self):
        """Run one round of training."""
        # Train all clients
        train_loss = 0
        val_loss = 0
        val_score = {}
        for client in self.clients.values():
            # Train and validate
            client_train_loss = client.train(self.personal)
            # Weight metrics by client dataset size
            train_loss += client_train_loss * client.data.weight
        # Aggregate and distribute
        self.aggregate_models()
        self.distribute_global_model()
        return train_loss


    def run_local_metrics_calculation(self, use_personal_model: bool) -> pd.DataFrame:
        """
        Performs local model analysis (weights, gradients).
        """
        print(f"Client {self.data.site_id}: Running local metrics calculation...")
        state = self.get_client_state(use_personal_model)
        model_to_analyze = state.model
        criterion = state.criterion

        # Sample one batch from the training loader
        try:
            data_iter = iter(self.data.train_loader)
            data_batch = next(data_iter)
            # Ensure batch is a tuple (features, labels)
            if not isinstance(data_batch, (list, tuple)) or len(data_batch) != 2:
                 raise ValueError("Dataloader must yield tuples of (features, labels)")

        except StopIteration:
            print(f"Warning: Client {self.data.site_id} train_loader is empty. Cannot calculate metrics.")
            return pd.DataFrame()
        except Exception as e:
             print(f"Error getting data batch for client {self.data.site_id}: {e}")
             return pd.DataFrame()


        metrics_df = calculate_local_layer_metrics(
            model=model_to_analyze,
            data_batch=data_batch,
            criterion=criterion,
            device=self.device,
        )
        print(f"Client {self.data.site_id}: Local metrics calculation finished.")
        return metrics_df

    def run_activation_extraction(self, probe_data_batch: Union[Tensor, Tuple], use_personal_model: bool) -> List[Tuple[str, np.ndarray]]:
        """
        Extracts model activations using the server-provided probe data batch.
        """
        print(f"Client {self.data.site_id}: Running activation extraction...")
        state = self.get_client_state(use_personal_model)
        model_to_analyze = state.model

        activations = get_model_activations(
            model=model_to_analyze,
            probe_data_batch=probe_data_batch,
            device=self.device,
            site_id=self.data.site_id # Pass site_id for storage in hook
        )
        print(f"Client {self.data.site_id}: Activation extraction finished.")
        return activations

