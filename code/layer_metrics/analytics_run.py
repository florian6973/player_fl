import argparse
import sys
import os
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
CODE_DIR = os.path.join(ROOT_DIR, 'code')
if CODE_DIR not in sys.path:
     sys.path.append(CODE_DIR)
from analytics_pipeline import AnalyticsConfig, AnalyticsExperiment
from configs import *
from helper import get_parameters_for_dataset # To get default runs

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Analytics Pipeline")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR, FMNIST).")
    parser.add_argument("-r", "--runs", type=int, default=None,
                        help="Number of independent runs. Overrides dataset default.")
    parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help="Base directory to save results.")

    args = parser.parse_args()

    # Determine number of runs
    if args.runs is None:
        default_params = get_parameters_for_dataset(args.dataset)
        # Use 'runs_analytics' if defined in defaults, else 'runs', else 1
        num_runs = 5
        print(f"Number of runs not specified, using default for {args.dataset}: {num_runs}")
    else:
        num_runs = args.runs
        print(f"Number of runs specified: {num_runs}")

    # Create configuration object using the analytics-specific config class
    config = AnalyticsConfig(
        dataset=args.dataset,
        num_runs=num_runs,
        results_dir=args.results_dir
    )

    # Instantiate and run the experiment using the refactored class
    experiment = AnalyticsExperiment(config)
    experiment.run_experiment()

    print(f"\nAnalytics pipeline finished for dataset: {args.dataset}.")

if __name__ == "__main__":
    main()