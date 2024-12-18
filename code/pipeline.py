ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'

import sys
import os
import numpy as np
import torch
import torch.nn as nn
sys.path.append(f'{ROOT_DIR}/code')
from data_processing import *
from servers import *
from helper import *
from losses import *
import models as ms
from helper import *
from performance_logging import *
import pickle
from config import *
import time


class ExperimentType:
    LEARNING_RATE = 'learning_rate'
    EVALUATION = 'evaluation'

class ExperimentConfig:
    def __init__(self, dataset, experiment_type, params_to_try=None):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.params_to_try = params_to_try or self._get_params_test()

    def _get_params_test(self):
        if self.experiment_type == ExperimentType.LEARNING_RATE:
            return LEARNING_RATES_TRY[self.dataset]
        else:
            return None

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

    def get_best_parameters(self, param_type, server_type, cost):
        """Get best hyperparameter value for given server type and cost."""
        results = self.load_results(param_type)
        if results is None or cost not in results:
            return None
        
        cost_results = results[cost]  # Now gives us lr-level dict
        
        # Collect metrics across all learning rates for this server
        server_metrics = {}
        for lr in cost_results.keys():
            if server_type not in cost_results[lr]:
                continue
            server_metrics[lr] = cost_results[lr][server_type]
        
        if not server_metrics:
            return None

        return self._select_best_hyperparameter(server_metrics)

    def _select_best_hyperparameter(self, lr_results):
        """Select best hyperparameter based on minimum loss."""
        best_loss = float('inf')
        best_param = None
        
        for lr, metrics in lr_results.items():
            avg_loss = np.median(metrics['global']['losses'])
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_param = lr
                
        return best_param


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_manager = ResultsManager(root_dir=ROOT_DIR, dataset=self.config.dataset, experiment_type = self.config.experiment_type)
        self.logger = performance_logger.get_logger(self.config.dataset, 'experiment')

    def run_experiment(self, costs):
        if self.config.experiment_type == ExperimentType.EVALUATION:
            return self._run_final_evaluation(costs)
        else:
            return self._run_hyperparameter_tuning(costs)
            
    def _check_existing_results(self, costs):
        """Check existing results and return remaining work to be done."""
        results = self.results_manager.load_results(self.config.experiment_type)
        remaining_costs = costs
        completed_runs = 0
        
        if results is not None:
            # Check which costs have been completed
            completed_costs = set(results.keys())
            remaining_costs = list(set(costs) - completed_costs)
            
            # Check number of completed runs if any results exist
            if completed_costs: 
                # Check the first cost that was completed to determine number of runs
                first_cost = next(iter(completed_costs))
                # Count number of elements in any metric list to determine completed runs
                first_param = next(iter(results[first_cost].keys()))
                first_server = next(iter(results[first_cost][first_param].keys()))
                completed_runs = len(results[first_cost][first_param][first_server]['global']['losses'])
                
        self.logger.info(f"Found {completed_runs} completed runs")
        
        remaining_runs = self.default_params['runs_lr'] - completed_runs
        if remaining_runs > 0:
            remaining_costs = costs
        self.logger.info(f"Remaining costs to process: {remaining_costs}")
        
        return results, remaining_costs, completed_runs

    def _run_hyperparameter_tuning(self, costs):
        """Run LR or Reg param tuning with multiple runs"""
        results, remaining_costs, completed_runs = self._check_existing_results(costs)
        
        # If no costs remain and all runs are completed, return existing results
        if not remaining_costs and completed_runs >= self.default_params['runs_lr']:
            self.logger.info("All experiments are already completed")
            return results
            
        # Calculate remaining runs
        remaining_runs = self.default_params['runs_lr'] - completed_runs
        
        for run in range(remaining_runs):
            current_run = completed_runs + run + 1
            self.logger.info(f"Starting run {current_run}/{self.default_params['runs_lr']}")
            results_run = {}
            
            for cost in remaining_costs:
                if self.config.experiment_type == ExperimentType.LEARNING_RATE:
                    hyperparams_list = [{'learning_rate': lr} for lr in self.config.params_to_try]
                    server_types = ['local', 'fedavg', 'pfedme', 'ditto']
                else:  # REG_PARAM
                    hyperparams_list = [{'reg_param': reg} for reg in self.config.params_to_try]
                    server_types = ['pfedme', 'ditto']

                tracking = {}
                for hyperparams in hyperparams_list:
                    param = next(iter(hyperparams.values()))
                    tracking[param] = self._hyperparameter_tuning(cost, hyperparams, server_types)
                results_run[cost] = tracking
            
            results = self.results_manager.append_or_create_metric_lists(results, results_run)
            self.results_manager.save_results(results, self.config.experiment_type)
            
        return results
    
    @log_execution_time
    def _hyperparameter_tuning(self, cost, hyperparams, server_types):
        """Run hyperparameter tuning for specific parameters."""
        self.logger.info(f"Starting hyperparameter tuning for cost: {cost}")
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'], cost)
        tracking = {}
        
        for server_type in server_types:
            self.logger.info(f"Training {server_type} model with hyperparameters: {hyperparams}")
            start_time = time.time()
            
            lr = hyperparams.get('learning_rate')
            config = self._create_trainer_config(lr)

            server = self._create_server_instance(server_type, config, cost, tuning = True)
            self._add_clients_to_server(server, client_dataloaders)
            metrics = self._train_and_evaluate(server, config.rounds)

            tracking[server_type] = metrics
            
            duration = time.time() - start_time
            self.logger.info(f"Completed {server_type} training in {duration:.2f}s")

        return tracking

    def _run_final_evaluation(self, costs):
        """Run final evaluation with multiple runs"""
        results = {}
        diversities = {}
        for run in range(self.default_params['runs']):
            try:
                print(f"Starting run {run + 1}/{self.default_params['runs']}")
                results_run = {}
                diversities_run = {}
                
                for cost in costs:
                    experiment_results = self._final_evaluation(cost)
                    results_run[cost] = experiment_results
                        
                results = self.results_manager.append_or_create_metric_lists(results, results_run)
                diversities = self.results_manager.append_or_create_metric_lists(diversities, diversities_run)
                self.results_manager.save_results(results, self.config.experiment_type)
                
            except Exception as e:
                print(f"Run {run + 1} failed with error: {e}")
                if results is not None:
                    self.results_manager.save_results(results,  self.config.experiment_type)
        
        return results, diversities


    def _final_evaluation(self, cost):
        tracking = {}
        server_types = ['local', 'fedavg', 'pfedme', 'ditto']
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'], cost)

        for server_type in server_types:
            print(f"Evaluating {server_type} model with best hyperparameters")
            lr = self.results_manager.get_best_parameters(
                ExperimentType.LEARNING_RATE, server_type, cost)
            print(lr, flush = True)
            if server_type in ['pfedme', 'ditto']:
                reg_param = self.results_manager.get_best_parameters(
                    ExperimentType.REG_PARAM, server_type, cost)
                config = self._create_trainer_config(lr, personalization_params={"reg_param": reg_param})
            else:
                config = self._create_trainer_config(lr)

            server = self._create_server_instance(server_type, config, cost, tuning = False)
            self._add_clients_to_server(server, client_dataloaders)
            metrics = self._train_and_evaluate(server, config.rounds)
            tracking[server_type] = metrics

        return tracking
    
    
    def _initialize_experiment(self, batch_size, cost):
        preprocessor = DataPreprocessor(self.config.dataset, batch_size)
        client_data = {}
        client_ids = self._get_client_ids(cost)
        
        for client_id in client_ids:
            client_num = int(client_id.split('_')[1])
            X, y = self._load_data(client_num, cost)
            client_data[client_id] = {'X': X, 'y': y}
        
        return preprocessor.process_clients(client_data)
    
    def _get_client_ids(self, cost):
        CLIENT_NUMS = {'IXITiny': 3, 'ISIC': 4}
        if self.config.dataset in CLIENT_NUMS and cost == 'all':
            CLIENT_NUM = CLIENT_NUMS[self.config.dataset]
        else:
            CLIENT_NUM = 2
        return [f'client_{i}' for i in range(1, CLIENT_NUM + 1)]
    
    def _create_trainer_config(self, learning_rate, personalization_params = None):
        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=5,
            rounds=self.default_params['rounds'],
            personalization_params=personalization_params
        )

    def _create_model(self, cost, learning_rate):
        if self.config.dataset in ['EMNIST', 'CIFAR']:
            with open(f'{self.data_dir}/CLASSES', 'rb') as f:
                classes_used = pickle.load(f)
            classes = len(set(classes_used[cost][0] + classes_used[cost][1]))
            model = getattr(ms, self.config.dataset)(classes)
        else:
            model = getattr(ms, self.config.dataset)()

        criterion = {
            'Synthetic': nn.BCELoss(),
            'Credit': nn.BCELoss(),
            'Weather': nn.MSELoss(),
            'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            'IXITiny': get_dice_loss,
            'ISIC': nn.CrossEntropyLoss()
        }.get(self.config.dataset, None)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=True,
            weight_decay=1e-4
        )
        return model, criterion, optimizer

    def _create_server_instance(self, server_type, config, cost, tuning):
        learning_rate = config.learning_rate
        model, criterion, optimizer = self._create_model(cost, learning_rate)
        globalmodelstate = ModelState(
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )

        server_mapping = {
            'local': Server,
            'fedavg': FedAvgServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer
        }

        server_class = server_mapping[server_type]
        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning)
        return server

    def _add_clients_to_server(self, server, client_dataloaders):
        is_personalized = server.server_type in ['pfedme', 'ditto']
        for client_id in client_dataloaders:
            if client_id == 'client_joint' and server.server_type != 'local':
                continue  # Skip this iteration
            else:
                clientdata = self._create_site_data(client_id, client_dataloaders[client_id])
                server.add_client(clientdata=clientdata, personal=is_personalized)

    def _create_site_data(self, client_id, loaders):
        return SiteData(
            site_id=client_id,
            train_loader=loaders[0],
            val_loader=loaders[1],
            test_loader=loaders[2]
        )

    def _load_data(self, client_num, cost):
        return loadData(self.config.dataset, f'{self.data_dir}', client_num, cost)

    @log_execution_time
    def _train_and_evaluate(self, server, rounds):
        eval_type = "validation" if server.tuning else "test"
        self.logger.info(f"Starting training for {rounds} rounds for server type {server.server_type}")
        
        for round_num in range(rounds):
            round_start = time.time()
            server.train_round()
            round_time = time.time() - round_start
            
            # Log every 20% of rounds
            if round_num % max(1, rounds // 5) == 0:
                self.logger.info(f"Completed round {round_num + 1}/{rounds} in {round_time:.2f}s")
                
                # Get current metrics
                state = server.global_site.state.global_state
                current_loss = state.val_losses[-1]
                current_score = state.val_scores[-1]
                self.logger.info(f"Current {eval_type} metrics - Loss: {current_loss:.4f}; Score: {current_score:4f}")
        
        if not server.tuning:
            # Final evaluation
            self.logger.info("Running final evaluation")
            server.test_global()
            state = server.global_site.state.global_state
        
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

        if server.server_type == 'fedavg' and not server.tuning:
            metrics['diversity'] = server.weight_diversities
        
        
        # Log per-client metrics
        for client_id, client in server.clients.items():
            state = (client.site.state.personal_state 
                    if client.site.state.personal_state is not None 
                    else client.site.state.global_state)
            
            if server.tuning:
                losses, scores = state.val_losses, state.val_scores 
            else:
                losses, scores = state.test_losses, state.test_scores 
                
            metrics['sites'][client_id] = {
                'losses': losses,
                'scores': scores
            }
            
            self.logger.info(f"Client {client_id} final metrics - "
                            f"Loss: {losses[-1]:.4f}, Score: {scores[-1]:.4f}")
    
        return metrics
