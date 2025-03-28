ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from helper import move_to_device, cleanup_gpu 
from configs import *
from clients import SiteData, ModelState
from analytics_clients import TrainerConfig, AnalyticsClient 
from servers import Server
from layer_analytics  import *


class AnalyticsServer(Server): # Inherit from Server or another specific server type
    """
    Extends a Federated Learning Server to orchestrate model analysis across clients.
    """
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        super().__init__(config, globalmodelstate)
        self.analysis_results = {'local': {}, 'similarity': {}} # Store results here
        self.probe_data_batch = None # Store the consistent batch for similarity
        self.num_cpus_analysis = config.num_cpus if hasattr(config, 'num_cpus') else 4 # Get CPUs from config

    # --- Override Client Creation ---
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool) -> AnalyticsClient:
        """
        Override to create AnalyticsClient instances instead of base Client.
        Make sure the base class you inherit from (`FLServer` here) also uses `_create_client`.
        """
        print(f"AnalyticsServer: Creating AnalyticsClient for site {clientdata.site_id}")
        # Ensure MetricsCalculator is available or passed correctly
        # Assuming MetricsCalculator is accessible via config or created here

        return AnalyticsClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Give each client a copy of the initial state
            metrics_calculator=None,
            personal_model=False
        )

    # --- Analytics Orchestration ---
    def _prepare_probe_data(self, num_samples_per_site: int = 16):
        """
        Creates a consistent data batch for activation similarity analysis by
        sampling from each client's training data. Should only be called once.
        """
        if self.probe_data_batch is not None:
            return # Already prepared

        print("Server: Preparing probe data batch for similarity analysis...")
        if not self.clients:
            print("Warning: No clients available to prepare probe data.")
            return

        all_features_list = []
        all_labels_list = []
        feature_type = None # To handle tuples vs tensors

        total_samples_needed = num_samples_per_site * len(self.clients)
        samples_collected = 0

        for client_id, client in self.clients.items():
            try:
                # Make sure dataloader is not empty
                if len(client.data.train_loader.dataset) == 0:
                     print(f"Warning: Client {client_id} dataset is empty. Skipping for probe data.")
                     continue

                data_iter = iter(client.data.train_loader)
                features, labels = next(data_iter)

                # Determine feature type from first client
                if feature_type is None:
                    feature_type = type(features)

                # Take specified number of samples
                num_samples = min(num_samples_per_site, len(labels))
                if num_samples == 0: continue

                if feature_type == tuple:
                    # Handle tuple features (e.g., input_ids, attention_mask)
                    sampled_features = tuple(f[:num_samples] for f in features)
                elif feature_type == torch.Tensor:
                    sampled_features = features[:num_samples]
                else:
                    print(f"Warning: Unsupported feature type {feature_type} for client {client_id}. Skipping.")
                    continue

                sampled_labels = labels[:num_samples]

                all_features_list.append(sampled_features)
                all_labels_list.append(sampled_labels)
                samples_collected += num_samples

            except StopIteration:
                print(f"Warning: Client {client_id} train_loader exhausted early. Skipping for probe data.")
                continue
            except Exception as e:
                print(f"Error getting probe data from client {client_id}: {e}")
                continue

        if not all_labels_list:
             print("Error: Could not collect any probe data samples.")
             return

        # Combine samples into a single batch
        combined_labels = torch.cat(all_labels_list, dim=0)

        if feature_type == tuple:
            # Combine tuple features element-wise
            num_feature_elements = len(all_features_list[0])
            combined_features_tuple = []
            for i in range(num_feature_elements):
                combined_features_tuple.append(torch.cat([f[i] for f in all_features_list], dim=0))
            combined_features = tuple(combined_features_tuple)
        elif feature_type == torch.Tensor:
            combined_features = torch.cat(all_features_list, dim=0)
        else:
            print("Error: Could not combine features due to unsupported type.")
            return


        self.probe_data_batch = (combined_features, combined_labels)
        print(f"Server: Probe data batch created with {len(combined_labels)} samples.")


    def run_analysis(self, round_identifier: str):
        """
        Orchestrates the analysis (local metrics and similarity) across all clients
        for a given round identifier (e.g., 'initial', 'final', 'round_10').
        """
        if not self.clients:
            print("Server: No clients to run analysis on.")
            return

        print(f"\n--- Server: Starting Analysis for '{round_identifier}' ---")
        start_time = time.time()

        # Determine if personal models should be used for analysis
        # Based on server's config or potentially the algorithm state
        use_personal = self.personal # Use the server's 'personal' flag

        # 1. Prepare Probe Data (if not already done)
        self._prepare_probe_data()

        # 2. Local Metrics Calculation
        print("Server: Requesting local metrics from clients...")
        current_local_metrics = {}
        for client_id, client in self.clients.items():
            try:
                metrics_df = client.run_local_metrics_calculation(
                    use_personal_model=use_personal,
                )
                current_local_metrics[client_id] = metrics_df
            except Exception as e:
                print(f"Error getting local metrics from client {client_id}: {e}")
                current_local_metrics[client_id] = pd.DataFrame() # Store empty df on error

        self.analysis_results['local'][round_identifier] = current_local_metrics
        print("Server: Local metrics collected.")

        # 3. Activation Similarity Calculation
        if self.probe_data_batch is None:
            print("Server: Skipping similarity calculation - probe data not available.")
        else:
            print("Server: Requesting activations from clients...")
            all_client_activations: Dict[str, List[Tuple[str, np.ndarray]]] = {}
            for client_id, client in self.clients.items():
                try:
                    # Ensure probe batch is on correct device or handle in client/analytics
                    activations = client.run_activation_extraction(
                        probe_data_batch=self.probe_data_batch,
                        use_personal_model=use_personal
                    )
                    all_client_activations[client_id] = activations
                except Exception as e:
                    print(f"Error getting activations from client {client_id}: {e}")
                    all_client_activations[client_id] = [] # Store empty list on error


            print("Server: Calculating activation similarity...")
            # Filter out clients that failed activation extraction
            valid_activations = {cid: act for cid, act in all_client_activations.items() if act}

            if len(valid_activations) >= 2:
                 similarity_results = calculate_activation_similarity(
                    activations_dict=valid_activations,
                    probe_data_batch=self.probe_data_batch,
                    num_sites=len(valid_activations),
                    cpus=self.num_cpus_analysis
                 )
                 self.analysis_results['similarity'][round_identifier] = similarity_results
                 print("Server: Activation similarity calculated.")
            else:
                 print("Server: Skipping similarity calculation - less than 2 clients provided valid activations.")
                 self.analysis_results['similarity'][round_identifier] = {}


        end_time = time.time()
        print(f"--- Server: Analysis for '{round_identifier}' finished ({end_time - start_time:.2f}s) ---")


    def save_analysis_results(self, filepath: str):
        """Saves the collected analysis results to a pickle file."""
        print(f"Server: Saving analysis results to {filepath}...")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.analysis_results, f)
            print("Server: Analysis results saved successfully.")
        except Exception as e:
            print(f"Error saving analysis results: {e}")



class AnalyticsFLServer(AnalyticsServer):
    """Base federated learning server with FedAvg implementation."""
    def aggregate_models(self):
        """Standard FedAvg aggregation."""
        # Reset global model parameters
        for param in self.serverstate.model.parameters():
            param.data.zero_()
            
        # Aggregate parameters
        for client in self.clients.values():
            client_model = client.personal_state.model if self.personal else client.global_state.model
            for g_param, c_param in zip(self.serverstate.model.parameters(), client_model.parameters()):
                g_param.data.add_(c_param.data * client.data.weight)

    def distribute_global_model(self):
        """Distribute global model to all clients."""
        global_state = self.serverstate.model.state_dict()
        for client in self.clients.values():
            client.set_model_state(global_state)

class AnalyticsFedAvgServer(AnalyticsFLServer):
    """FedAvg server implementation."""
    pass