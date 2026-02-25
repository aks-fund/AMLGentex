import argparse
import os
import sys
from time import time

import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_creation import DataGenerator
from src.data_tuning import DataTuner
from src.feature_engineering import DataPreprocessor
from src.utils.logging import configure_logging


def main():
    experiment = 'template_experiment'

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=False, default=f'experiments/{experiment}')
    parser.add_argument('--num_trials_data', type=int, default=10, help='Number of outer data optimization trials')
    parser.add_argument('--num_trials_model', type=int, default=15, help='Number of inner model trials per data trial')
    parser.add_argument('--model', type=str, default='DecisionTreeClassifier',
                        help='Model to use for evaluation')
    parser.add_argument('--optimization_mode', type=str, default='knowledge_free',
                        choices=['knowledge_free', 'operational'],
                        help='knowledge_free = paper FPR-target mode, operational = constrained utility mode')
    parser.add_argument('--target_fpr', type=float, default=0.98,
                        help='Target FPR for knowledge_free mode (paper uses ~0.98)')
    parser.add_argument('--fpr_threshold', type=float, default=0.5,
                        help='Probability threshold used when computing FPR in knowledge_free mode')
    parser.add_argument('--constraint_type', type=str, default='K', choices=['K', 'fpr', 'recall'],
                        help='Constraint type for operational mode')
    parser.add_argument('--constraint_value', type=float, default=100.0,
                        help='Constraint value for operational mode')
    parser.add_argument('--utility_metric', type=str, default='precision', choices=['precision', 'recall'],
                        help='Utility metric for operational mode')
    parser.add_argument('--target', type=float, default=0.8,
                        help='Target value for operational utility metric')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-update-config', action='store_true',
                        help='Skip updating data.yaml with selected best trial')
    args = parser.parse_args()
    if not 0.0 <= args.target_fpr <= 1.0:
        raise ValueError(f"--target_fpr must be in [0, 1], got {args.target_fpr}")

    args.data_conf_file = f'{args.experiment_dir}/config/data.yaml'
    args.preprocessing_config = f'{args.experiment_dir}/config/preprocessing.yaml'
    args.models_config = f'{args.experiment_dir}/config/models.yaml'
    args.bo_dir = f'{args.experiment_dir}/data_tuning'

    if not os.path.isabs(args.data_conf_file):
        args.data_conf_file = os.path.abspath(args.data_conf_file)
    if not os.path.isabs(args.preprocessing_config):
        args.preprocessing_config = os.path.abspath(args.preprocessing_config)
    if not os.path.isabs(args.models_config):
        args.models_config = os.path.abspath(args.models_config)
    if not os.path.isabs(args.bo_dir):
        args.bo_dir = os.path.abspath(args.bo_dir)

    os.makedirs(args.bo_dir, exist_ok=True)
    log_file = os.path.join(args.bo_dir, 'tune_data.log')
    configure_logging(verbose=False, log_file=log_file)

    from src.utils.config import load_data_config, load_preprocessing_config

    data_config = load_data_config(args.data_conf_file)
    with open(args.data_conf_file, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    configure_logging(verbose=True, log_file=log_file)
    generator = DataGenerator(args.data_conf_file)

    print("Running spatial simulation (graph generation) once...")
    print("This graph will be reused across all optimization trials.\n")
    generator.run_spatial(force=False)

    preprocessing_config = load_preprocessing_config(args.preprocessing_config)
    if args.optimization_mode == 'knowledge_free' and preprocessing_config.get('num_windows', 1) < 4:
        print(
            f"knowledge_free mode: increasing num_windows "
            f"{preprocessing_config.get('num_windows', 1)} -> 4 to match paper-style feature richness."
        )
        preprocessing_config['num_windows'] = 4
    preprocessor = DataPreprocessor(preprocessing_config)

    configure_logging(verbose=False, log_file=log_file)

    with open(args.models_config, 'r') as f:
        models_config = yaml.safe_load(f)

    if args.model not in models_config:
        raise ValueError(
            f"Model '{args.model}' not found in {args.models_config}. "
            f"Available: {list(models_config.keys())}"
        )

    if args.optimization_mode == 'knowledge_free' and args.model != 'DecisionTreeClassifier':
        print(
            "knowledge_free mode is paper-calibrated for DecisionTreeClassifier. "
            f"Overriding model '{args.model}' -> 'DecisionTreeClassifier'."
        )
        args.model = 'DecisionTreeClassifier'

    combined_config = {
        'preprocess': preprocessing_config.copy(),
        **models_config,
    }
    combined_config['preprocess']['preprocessed_data_dir'] = os.path.join(args.bo_dir, 'preprocessed')

    print("=" * 60)
    print("Data Tuning Configuration")
    print("=" * 60)
    print(f"Mode: {args.optimization_mode}")
    print(f"Model: {args.model}")
    if args.optimization_mode == 'knowledge_free':
        print(f"Objective: Minimize |FPR - {args.target_fpr}| + feature-importance dispersion")
        print(f"FPR threshold for scoring: {args.fpr_threshold}")
    elif args.constraint_type == 'K':
        print(f"Objective: Optimize {args.utility_metric} in top {int(args.constraint_value)} alerts")
        print(f"Target: {args.target:.1%} {args.utility_metric}")
    elif args.constraint_type == 'fpr':
        print(f"Objective: Optimize {args.utility_metric} at FPR <= {args.constraint_value}")
        print(f"Target: {args.target:.1%} {args.utility_metric}")
    elif args.constraint_type == 'recall':
        print(f"Objective: Optimize {args.utility_metric} at Recall >= {args.constraint_value}")
        print(f"Target: {args.target:.1%} {args.utility_metric}")
    print(f"Data trials: {args.num_trials_data}")
    print(f"Model trials per data trial: {args.num_trials_model}")
    print(f"Random seed: {args.seed}")
    print("=" * 60 + "\n")

    tuner = DataTuner(
        data_conf_file=args.data_conf_file,
        config=combined_config,
        generator=generator,
        preprocessor=preprocessor,
        target=args.target,
        constraint_type=args.constraint_type,
        constraint_value=args.constraint_value,
        utility_metric=args.utility_metric,
        model=args.model,
        bo_dir=args.bo_dir,
        seed=args.seed,
        num_trials_model=args.num_trials_model,
        optimization_mode=args.optimization_mode,
        target_fpr=args.target_fpr,
        fpr_threshold=args.fpr_threshold,
    )

    start = time()
    best_trials = tuner(args.num_trials_data)
    elapsed = time() - start
    print(f'\nTotal execution time: {elapsed/60:.1f} minutes\n')

    if not args.no_update_config and best_trials:
        import math

        if args.optimization_mode == 'knowledge_free':
            # In paper mode, objective 1 is already FPR deviation from target.
            best_trial = min(best_trials, key=lambda t: t.values[0])
        else:
            # Operational mode balances utility and feature-importance objectives.
            best_trial = min(best_trials, key=lambda t: math.sqrt(sum(v**2 for v in t.values)))

        tuner.optimizer.update_config_with_trial(best_trial.number)


if __name__ == '__main__':
    main()
