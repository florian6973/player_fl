from configs import *
from helper import *
from dataset_processing import *
import models
from losses import MulticlassFocalLoss
from clients import *
from servers import *
from performance_logging import *
import time
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
        return os.path.join(self.root_dir,'results', 
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
        """Select best hyperparameter based on minimum median of final losses."""
        best_loss = float('inf')
        best_param = None
        
        for lr, metrics in lr_results.items():
            # Extract final loss from each run
            final_losses = [run[-1] for run in metrics['global']['losses']]
            median_loss = np.median(final_losses)
            
            if median_loss < best_loss:
                best_loss = median_loss
                best_param = lr
        
        return best_param


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.root_dir = ROOT_DIR
        self.results_manager = ResultsManager(root_dir=ROOT_DIR, dataset=self.config.dataset, experiment_type = self.config.experiment_type)
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        self.logger = performance_logger.get_logger(self.config.dataset, self.config.experiment_type) #Import instantiated object from performance_logging.py

    def run_experiment(self):
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation()
        else:
            return self._run_hyperparameter_tuning()
            
    def _check_existing_results(self, server_types):
        """
        Check existing results and return remaining work to be done for each server type.
        Returns a dictionary of completed runs per server type and the loaded results.
        """
        results = self.results_manager.load_results(self.config.experiment_type)
        completed_runs = {server_type: 0 for server_type in server_types}
        
        if results is not None:
            # Check completion status for each parameter and server type
            for param in results:
                for server_type in server_types:
                    if server_type in results[param]:
                        # Count completed runs for this server type
                        runs = len(results[param][server_type]['global']['losses'])
                        completed_runs[server_type] = max(completed_runs[server_type], runs)
        return results, completed_runs

    def _run_hyperparameter_tuning(self):
        """Run LR or Reg param tuning with multiple runs, tracking per server type"""
        self.logger.info("Starting hyperparameter tuning experiment")
        server_types = ALGORITHMS
        if self.config.experiment_type == ExperimentType.LEARNING_RATE:
            hyperparams_list = [{'learning_rate': lr} for lr in self.default_params['learning_rates_try']]
            self.logger.info(f"Testing learning rates: {self.default_params['learning_rates_try']}")
            
        results, completed_runs = self._check_existing_results(server_types)
        total_runs = self.default_params['runs_lr']
        
        # Continue until all server types have completed all runs
        while min(completed_runs.values()) < total_runs:
            results_run = {}
            current_run = min(completed_runs.values()) + 1
            self.logger.info(f"Starting Run {current_run}/{total_runs}")
            for hyperparams in hyperparams_list:
                param = next(iter(hyperparams.values()))
                self.logger.info(f"Starting tuning for: {param}")
                results_run[param] = {}
                
                # Only run for server types that need more runs
                remaining_server_types = [
                    server_type for server_type in server_types 
                    if completed_runs[server_type] < total_runs
                ]
                
                if remaining_server_types:
                    results_run[param].update(
                        self._hyperparameter_tuning(hyperparams, remaining_server_types)
                    )
            
            # Update results and completion tracking
            results = self.results_manager.append_or_create_metric_lists(results, results_run)
            self.results_manager.save_results(results, self.config.experiment_type)
            
            # Update completion status
            for server_type in server_types:
                if server_type in remaining_server_types:
                    completed_runs[server_type] += 1
        
        return results
    
    def _hyperparameter_tuning(self, hyperparams, server_types):
        """Run hyperparameter tuning for specific parameters."""
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'])
        tracking = {}
        
        for server_type in server_types:
            self.logger.info(f"Starting server type: {server_type}")
            try:
                # Create and run server
                server = self._create_server_instance(server_type, hyperparams, tuning=True)
                self._add_clients_to_server(server, client_dataloaders)
                metrics = self._train_and_evaluate(server, server.config.rounds)
                tracking[server_type] = metrics
                
            finally:
                del server
                cleanup_gpu()
        
        return tracking

    def _run_final_evaluation(self):
        """Run final evaluation with multiple runs"""
        self.logger.info("Starting final evaluation phase")
        results = {}
        for run in range(self.default_params['runs']):
            try:
                self.logger.info(f"Starting run {run + 1}/{self.default_params['runs']}")
                results_run = self._final_evaluation()
                results = self.results_manager.append_or_create_metric_lists(results, results_run)
                self.results_manager.save_results(results, self.config.experiment_type)
                self.logger.info(f"Successfully completed run {run + 1}")
                
            except Exception as e:
                self.logger.error(f"Run {run + 1} failed with error: {str(e)}")
                if results is not None:
                    self.results_manager.save_results(results,  self.config.experiment_type)
        
        return results


    def _final_evaluation(self):
        tracking = {}
        server_types = ALGORITHMS
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'])

        for server_type in server_types:
            self.logger.info(f"Evaluating {server_type} model with best hyperparameters")
            print(f"Evaluating {server_type} model with best hyperparameters")
            try:
                lr = self.results_manager.get_best_parameters(
                    ExperimentType.LEARNING_RATE, server_type)
                hyperparms = {'learning_rate': lr}
                server = self._create_server_instance(server_type, hyperparms, tuning = False)
                self._add_clients_to_server(server, client_dataloaders)
                metrics = self._train_and_evaluate(server, server.config.rounds)
                tracking[server_type] = metrics
                self.logger.info(f"Completed {server_type} evaluation")
            finally:
                del server
                cleanup_gpu()
                
        return tracking
    
    
    def _initialize_experiment(self, batch_size):
        # Initialize preprocessor
        preprocessor = DataPreprocessor(self.config.dataset, batch_size)
        
        # Use UnifiedDataLoader to load the full dataset
        loader = UnifiedDataLoader(root_dir=self.root_dir, dataset_name=self.config.dataset)
        dataset_df = loader.load()  # Returns a DataFrame with 'data', 'label', and 'site'
        
        # Process client data into train, validation, and test loaders
        client_data = preprocessor.process_client_data(dataset_df)
        return client_data

    
    def _get_client_ids(self):
        return [f'client_{i}' for i in range(1, self.default_params['num_clients'] + 1)]
    
    def _create_trainer_config(self, server_type, learning_rate, algorithm_params = None):
        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=self.default_params['epochs_per_round'],
            rounds=self.default_params['rounds'],
            num_clients=self.default_params['num_clients'],
            requires_personal_model= True if server_type in ['pfedme', 'ditto'] else False,
            algorithm_params=algorithm_params
        )

    def _create_model(self, learning_rate):
        classes = self.default_params['classes']
        model = getattr(models, self.config.dataset)(classes)
        criterion = {'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            "FMNIST": nn.CrossEntropyLoss(),
            "ISIC": MulticlassFocalLoss(num_classes=classes, alpha = [0.87868852, 0.88131148, 0.82793443, 0.41206557], gamma = 1),
            "Sentiment": nn.CrossEntropyLoss(),
            "Heart": MulticlassFocalLoss(num_classes=classes, alpha = [0.12939189, 0.18108108, 0.22331081, 0.22364865, 0.24256757], gamma = 1),
            "mimic":MulticlassFocalLoss(num_classes=classes, alpha = [0.15,0.85], gamma = 1),
        }.get(self.config.dataset, None)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=1e-4
        )
        return model, criterion, optimizer

    def _create_server_instance(self, server_type, hyperparams, tuning):
        lr = hyperparams.get('learning_rate')
        algorithm_params = get_algorithm_config(server_type, self.config.dataset)
        config = self._create_trainer_config(server_type,lr, algorithm_params=algorithm_params)
        learning_rate = config.learning_rate
        
    
        # Update config with personalization parameters
        config.algorithm_params = algorithm_params
        model, criterion, optimizer = self._create_model(learning_rate)
        globalmodelstate = ModelState(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        server_mapping = {
            'local': Server,
            'fedavg': FedAvgServer,
            'fedprox': FedProxServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer,
            'localadaptation':LocalAdaptationServer,
            'babu':BABUServer,
            'fedlp':FedLPServer,
            'fedlama':FedLAMAServer,
            'pfedla':pFedLAServer,
            'layerpfl':LayerServer
        }

        server_class = server_mapping[server_type]
        server = server_class(config=config, globalmodelstate = globalmodelstate)
        server.set_server_type(server_type, tuning)
        return server

    def _add_clients_to_server(self, server, client_dataloaders):        
        for client_id, loaders in client_dataloaders.items():
            # Create SiteData from loaders
            client_data = self._create_site_data(client_id, loaders)
            server.add_client(clientdata=client_data)

    def _create_site_data(self, client_id, loaders):
        return SiteData(
            site_id=client_id,
            train_loader=loaders[0],
            val_loader=loaders[1],
            test_loader=loaders[2]
        )

    def _load_data(self, client_num):
        loader = UnifiedDataLoader(root_dir=self.root_dir, dataset_name=self.config.dataset)
        dataset_df = loader.load()
        
        # Filter dataset for the specific client (if applicable)
        client_df = dataset_df[dataset_df['site'] == client_num]
        return client_df['data'].values, client_df['label'].values


    def _train_and_evaluate(self, server, rounds):
        for round_num in range(rounds):
            server.train_round()        
            if (round_num +1 == rounds) and (server.server_type in ['localadaptation', 'babu']):
                server.train_round(final_round = True)
 
        if not server.tuning:
            # Final evaluation
            server.test_global()
        
        state = server.serverstate
        
        if server.tuning:
            losses, scores = state.val_losses, state.val_scores 
        else:
            losses, scores = state.test_losses, state.test_scores 
            
        metrics = {
            'global': {
                'losses': losses,
                'scores': scores
            },
            'sites': {}
        }
        
        # Log per-client metrics
        for client_id, client in server.clients.items():
            state = (client.personal_state 
                    if client.personal_state is not None 
                    else client.global_state)
            
            if server.tuning:
                losses, scores = state.val_losses, state.val_scores 
            else:
                losses, scores = state.test_losses, state.test_scores 
                
            metrics['sites'][client_id] = {
                'losses': losses,
                'scores': scores
            }
            
    
        return metrics


def check_client_models_identical(server, personal):
    """
    Check if all client models are identical.
    Returns: 
        bool: True if all models are identical, False otherwise
        dict: Differences found between models if any
    """
    clients = server.clients
    if not clients:
        return True, {}
    
    # Get first client's model state as reference
    first_client_id = list(clients.keys())[0]
    reference_state = clients[first_client_id].personal_state.model.state_dict() if personal else clients[first_client_id].global_state.model.state_dict()
    
    differences = {}
    models_identical = True
    
    # Compare each client's model with the reference
    for client_id, client in clients.items():
        if client_id == first_client_id:
            continue
            
        current_state = client.personal_state.model.state_dict() if personal else client.global_state.model.state_dict()
        
        # Compare each parameter
        for key in reference_state.keys():
            if not torch.equal(reference_state[key], current_state[key]):
                if client_id not in differences:
                    differences[client_id] = {}
                differences[client_id][key] = {
                    'max_diff': float(torch.max(torch.abs(reference_state[key] - current_state[key]))),
                    'mean_diff': float(torch.mean(torch.abs(reference_state[key] - current_state[key])))
                }
                models_identical = False
    
    return models_identical, differences
