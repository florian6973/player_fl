
from configs import ROOT_DIR, EVAL_DIR, RESULTS_DIR
import sys
sys.path.append(f'{ROOT_DIR}/code')
sys.path.append(f'{EVAL_DIR}')
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


def load_lr_results(DATASET):
    with open(f'{ROOT_DIR}/results/lr_tuning/{DATASET}_lr_tuning.pkl', 'rb') as f :
        results = pickle.load(f)

    return results

def get_lr_results(DATASET):
    results = load_lr_results(DATASET)
    results_output = []
    for lr, algorithms in results.items():
        for algo, metrics in algorithms.items():
            global_metrics = metrics['global']
            
            # For each round, calculate the mean loss across all runs
            num_rounds = len(global_metrics['losses'][0])  # Assuming all runs have same number of rounds
            median_losses_per_round = []
            accuracies_at_best = []
            f1_at_best = []
            
            for round_idx in range(num_rounds):
                # Get losses for this round across all runs
                round_losses = [run[round_idx] for run in global_metrics['losses']]
                median_loss = np.mean(round_losses)
                median_losses_per_round.append((round_idx, median_loss))
            
            # Find the round with the best (lowest) mean loss
            best_round_idx, best_median_loss = min(median_losses_per_round, key=lambda x: x[1])
            # Get accuracies and F1 scores from the best round
            accuracies = [run[best_round_idx]['accuracy'] for run in global_metrics['scores']]
            f1_scores = [run[best_round_idx]['f1_macro'] for run in global_metrics['scores']]
            
            results_output.append({
                'learning_rate': lr,
                'algorithm': algo,
                'best_round': best_round_idx,
                'median_loss': best_median_loss,
                'median_accuracy': np.mean(accuracies),
                'median_f1': np.mean(f1_scores)
            })

    df_results = pd.DataFrame(results_output)
    # Sort by best mean accuracy among the rounds with best mean loss for each algorithm
    return df_results.loc[df_results.groupby('algorithm')['median_loss'].idxmin()].sort_values(by='median_loss', ascending=True)


def load_eval_results(DATASET):
    with open(f'{RESULTS_DIR}/evaluation/{DATASET}_evaluation.pkl', 'rb') as f :
        results = pickle.load(f)
    return results

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, 
                          confidence: float = 0.95, min_samples: int = 2) -> tuple:
    """Calculate bootstrap confidence interval for the median using vectorized operations.
    """
    if len(data) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples for bootstrap CI")
        
    size = len(data)
    rng = np.random.default_rng()
    
    # Vectorized sampling and median calculation
    indices = rng.integers(0, size, size=(n_bootstrap, size))
    bootstrap_samples = data[indices]
    bootstrap_medians = np.median(bootstrap_samples, axis=1)
    
    lower_percentile = ((1 - confidence) / 2) * 100
    upper_percentile = (confidence + (1 - confidence) / 2) * 100
    
    return np.percentile(bootstrap_medians, [lower_percentile, upper_percentile])

class ResultAnalyzer:
    """Analyzes experimental results across multiple datasets and metrics."""
    
    def __init__(self, all_dataset_results: Dict):
        """Initialize analyzer with experimental results.
        """
        self.all_dataset_results = all_dataset_results        
        # Extract metrics from first valid dataset
        self.first_dataset = next(iter(all_dataset_results.values()))
        self.first_scores = self.first_dataset[next(iter(self.first_dataset.keys()))]['global']['scores'][0][0]
        self.metrics = list(self.first_scores.keys()) + ['loss']
        

    
    def analyze_dataset(self, results: Dict) -> Dict:
        """Analyze results for all metrics in a single dataset.
        """
        algorithms = list(results.keys())
        metrics_data = {}
        
        for algo in algorithms:
            metrics_data[algo] = {}
            
            # Get all scores and calculate statistics once
            scores_dict = {}
            for metric in self.metrics[:-1]:  # Exclude loss
                scores = np.array([x[0][metric] for x in results[algo]['global']['scores']])
                scores_dict[metric] = scores
            
            # Handle loss consistently - store raw values
            losses = np.array([x[0] for x in results[algo]['global']['losses']])
            scores_dict['loss'] = losses
            
            # Calculate statistics for all metrics
            for metric, scores in scores_dict.items():
                try:
                    median = np.median(scores)
                    ci_lower, ci_upper = bootstrap_ci(scores)
                    metrics_data[algo][metric] = {
                        'median': median,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    }
                except ValueError as e:
                    print(f"Warning: Could not compute CI for {algo}, {metric}: {str(e)}")
                    metrics_data[algo][metric] = {
                        'median': median,
                        'ci_lower': median,
                        'ci_upper': median
                    }
        
        return metrics_data
    
    def _get_client_metrics(self, results: Dict, algorithm: str, client_id: str) -> Dict:
        """Extract metrics for a specific client.
        """
        client_data = results[algorithm]['sites'][client_id]
        metrics = {}
        
        # Store raw loss values consistently
        loss_values = np.array([loss[0] for loss in client_data['losses']])
        metrics['loss'] = np.median(loss_values)
        
        # Process other metrics
        scores = client_data['scores']
        for metric in self.metrics[:-1]:
            values = np.array([score[0][metric] for score in scores])
            metrics[metric] = np.median(values)
        
        return metrics
    
    def analyze_fairness(self, results: Dict) -> pd.DataFrame:
        """Analyze fairness metrics for a single dataset.
        """
        # Keep the same logic for getting all algorithms and baselines
        algorithms = [algo for algo in results.keys() if algo not in ['local', 'fedavg']]
        client_ids = list(results['local']['sites'].keys())
        
        # Get baselines once
        baselines = {
            'local': {client_id: self._get_client_metrics(results, 'local', client_id)
                    for client_id in client_ids},
            'fedavg': {client_id: self._get_client_metrics(results, 'fedavg', client_id)
                    for client_id in client_ids}
        }
        
        fairness_data = []
        
        for algo in algorithms:  # This already excludes local and fedavg
            algo_metrics = {
                client_id: self._get_client_metrics(results, algo, client_id)
                for client_id in client_ids
            }
            
            for metric in self.metrics:
                metric_values = np.array([metrics[metric] for metrics in algo_metrics.values()])
                variance = np.var(metric_values)
                
                # For comparing with baseline, consider if higher or lower is better
                is_loss = metric == 'loss'
                if is_loss:
                    better_count = sum(
                        1 for client_id in client_ids
                        if (algo_metrics[client_id][metric] < min(
                            baselines['local'][client_id][metric],
                            baselines['fedavg'][client_id][metric]
                            )
                        )
                    )
                else: 
                    better_count = sum(
                        1 for client_id in client_ids 
                        if ( algo_metrics[client_id][metric] > max(
                            baselines['local'][client_id][metric],
                            baselines['fedavg'][client_id][metric]
                            )
                        )
                    )
                pct_better = (better_count / len(client_ids)) * 100
                
                fairness_data.append({
                    'Algorithm': algo,
                    'Metric': metric,
                    'Variance': variance,
                    'Pct_Better': pct_better
                })
        
        return pd.DataFrame(fairness_data)
    def analyze_all(self) -> Dict:
        """Analyze all datasets and metrics.
        """
        results = {
            'metrics': {},
            'fairness': {}
        }
        
        for dataset_name, dataset_results in self.all_dataset_results.items():
            # Analyze metrics
            metrics_data = self.analyze_dataset(dataset_results)
            for metric in self.metrics:
                if metric not in results['metrics']:
                    results['metrics'][metric] = []
                
                for algo, algo_data in metrics_data.items():
                    results['metrics'][metric].append({
                        'Dataset': dataset_name,
                        'Algorithm': algo,
                        'Median': algo_data[metric]['median'],
                        'CI_Lower': algo_data[metric]['ci_lower'],
                        'CI_Upper': algo_data[metric]['ci_upper']
                    })
            
            # Analyze fairness
            try:
                fairness_df = self.analyze_fairness(dataset_results)
                for _, row in fairness_df.iterrows():
                    if row['Metric'] not in results['fairness']:
                        results['fairness'][row['Metric']] = []
                    results['fairness'][row['Metric']].append({
                        'Dataset': dataset_name,
                        'Algorithm': row['Algorithm'],
                        'Variance': row['Variance'],
                        'Pct_Better': row['Pct_Better']
                    })
            except Exception as e:
                print(f"Warning: Fairness analysis failed for {dataset_name}: {str(e)}")
        
        # Convert to DataFrames
        for metric in results['metrics']:
            results['metrics'][metric] = pd.DataFrame(results['metrics'][metric])
        for metric in results['fairness']:
            results['fairness'][metric] = pd.DataFrame(results['fairness'][metric])
        
        return results
    
    def _perform_friedman_test(self, pivot_table: pd.DataFrame) -> Optional[Dict]:
        """Perform Friedman test on pivot table data.
        """
        data_columns = [col for col in pivot_table.columns if col not in ['Mean Rank', 'Friedman Test']]
        
        if len(data_columns) < 2:
            return None
            
        try:
            statistic, pvalue = stats.friedmanchisquare(
                *[pivot_table[col] for col in data_columns]
            )
            return {'statistic': statistic, 'pvalue': pvalue}
        except Exception:
            return None
    
    def format_tables(self, DATASETS, analysis_results: Dict) -> Dict:
        """Format analysis results into presentation-ready tables with confidence intervals."""
        formatted_tables = {}
        
        # Define custom ordering for algorithms and datasets
        algorithm_order = [
            'local', 'fedavg', 'fedprox', 'pfedme', 'ditto', 
            'localadaptation', 'babu', 'fedlp', 'fedlama', 
            'pfedla', 'layerpfl', 'layerpfl_random'
        ]
        
        dataset_order = DATASETS +  ['Mean Rank']

        fairness_algorithm_order = [
            'fedprox', 'pfedme', 'ditto', 'localadaptation', 'babu', 
            'fedlp', 'fedlama', 'pfedla', 'layerpfl', 'layerpfl_random'
        ]
        
        # Format metric summaries
        for metric, df in analysis_results['metrics'].items():
            # Create pivot tables for each statistic
            pivot_median = df.pivot(index='Algorithm', columns='Dataset', values='Median')
            pivot_ci_lower = df.pivot(index='Algorithm', columns='Dataset', values='CI_Lower')
            pivot_ci_upper = df.pivot(index='Algorithm', columns='Dataset', values='CI_Upper')
            
            # Calculate ranks - handle loss metric differently
            ranks = df.copy()
            ascending = metric == 'loss'  # True for loss (lower is better)
            ranks['Rank'] = df.groupby('Dataset')['Median'].rank(ascending=ascending)
            mean_ranks = ranks.groupby('Algorithm')['Rank'].mean()
            
            # Create combined pivot table
            pivot = pivot_median.copy()
            pivot['Mean Rank'] = mean_ranks
            
            # Format numbers as strings with confidence intervals
            for col in pivot.columns[:-1]:  # Exclude Mean Rank
                pivot[col] = pivot.apply(
                    lambda x: f"{x[col]:.3f} ± {(pivot_ci_upper[col][x.name] - pivot_ci_lower[col][x.name])/2:.3f}" 
                    if not pd.isna(x[col]) else "N/A",
                    axis=1
                )
            
            pivot['Mean Rank'] = pivot['Mean Rank'].apply(lambda x: f"{x:.2f}")
            
            # Reorder rows and columns according to defined orders
            pivot = pivot.reindex(index=algorithm_order)
            pivot = pivot[dataset_order]
            
            # Perform Friedman test if possible
            friedman_result = self._perform_friedman_test(pivot_median)
            formatted_tables[f'{metric}_summary'] = {
                'table': pivot,
                'friedman': f"Friedman test p-value: {friedman_result['pvalue']:.5f}" if friedman_result else "Friedman test not applicable"
            }
            
            # Format fairness summaries
            if 'fairness' in analysis_results:
                for metric, df in analysis_results['fairness'].items():
                    # Variance tables (lower is better)
                    var_pivot = df.pivot(index='Algorithm', columns='Dataset', values='Variance')
                    var_ranks = df.copy()
                    var_ranks['Rank'] = df.groupby('Dataset')['Variance'].rank()
                    var_mean_ranks = var_ranks.groupby('Algorithm')['Rank'].mean()
                    
                    var_pivot['Mean Rank'] = var_mean_ranks
                    
                    # Format numbers
                    for col in var_pivot.columns[:-1]:  # Exclude Mean Rank
                        var_pivot[col] = var_pivot[col].apply(lambda x: f"{x:.6f}")
                    var_pivot['Mean Rank'] = var_pivot['Mean Rank'].apply(lambda x: f"{x:.2f}")
                    
                    # Reorder rows and columns
                    var_pivot = var_pivot.reindex(index=fairness_algorithm_order)
                    var_pivot = var_pivot[dataset_order]
                    
                    # Add Friedman test result
                    friedman_result = self._perform_friedman_test(var_pivot)
                    formatted_tables[f'fairness_variance_{metric}'] = {
                        'table': var_pivot,
                        'friedman': f"Friedman test p-value: {friedman_result['pvalue']:.5f}" if friedman_result else "Friedman test not applicable"
                    }
                    
                    # Percentage better tables (higher is better)
                    pct_pivot = df.pivot(index='Algorithm', columns='Dataset', values='Pct_Better')
                    pct_ranks = df.copy()
                    pct_ranks['Rank'] = df.groupby('Dataset')['Pct_Better'].rank(ascending=False)
                    pct_mean_ranks = pct_ranks.groupby('Algorithm')['Rank'].mean()
                    
                    pct_pivot['Mean Rank'] = pct_mean_ranks
                    
                    # Format numbers
                    for col in pct_pivot.columns[:-1]:  # Exclude Mean Rank
                        pct_pivot[col] = pct_pivot[col].apply(lambda x: f"{x:.1f}")
                    pct_pivot['Mean Rank'] = pct_pivot['Mean Rank'].apply(lambda x: f"{x:.2f}")
                    
                    # Reorder rows and columns
                    pct_pivot = pct_pivot.reindex(index=fairness_algorithm_order)
                    pct_pivot = pct_pivot[dataset_order]
                    
                    # Add Friedman test result
                    friedman_result = self._perform_friedman_test(pct_pivot)
                    formatted_tables[f'fairness_pct_better_{metric}'] = {
                        'table': pct_pivot,
                        'friedman': f"Friedman test p-value: {friedman_result['pvalue']:.5f}" if friedman_result else "Friedman test not applicable"
                    }
        
        return formatted_tables

def analyze_experiment_results(DATASETS) -> Dict:
    """
    Analyze experimental results across all datasets.
    """
    try:
        all_results = {}
        for dataset in DATASETS:
            all_results[dataset] = load_eval_results(dataset)
        analyzer = ResultAnalyzer(all_results)
        analysis_results = analyzer.analyze_all()
        formatted_tables = analyzer.format_tables(DATASETS, analysis_results)
        return formatted_tables
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")
        return {}