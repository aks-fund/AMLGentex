import argparse
import os
import sys
import tempfile
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_creation import DataGenerator
from src.utils.config import load_data_config
from src.utils.logging import configure_logging
from time import time

def main(conf_file: str, verbose: bool = True):
    # Load config with auto-discovered paths
    config = load_data_config(conf_file)

    # Configure logging with log file in temporal output directory
    log_file = os.path.join(config['output']['directory'], 'generate.log')
    os.makedirs(config['output']['directory'], exist_ok=True)
    configure_logging(verbose=verbose, log_file=log_file)

    # Write temporary config file for DataGenerator (it expects a file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp, default_flow_style=False, sort_keys=False)
        tmp_config_path = tmp.name

    try:
        generator = DataGenerator(tmp_config_path)
        tx_log_file = generator()
        print(f'\nSynthetic AML data generated\n    Raw transaction log file: {tx_log_file}')
    finally:
        # Clean up temporary file
        os.unlink(tmp_config_path)

if __name__ == "__main__":
    EXPERIMENT = 'template_experiment'
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the data config file', default=f'experiments/{EXPERIMENT}/config/data.yaml')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress messages')
    args = parser.parse_args()
    if not os.path.isabs(args.conf_file):
        args.conf_file = os.path.abspath(args.conf_file)
    t = time()
    main(args.conf_file, verbose=args.verbose)
    t = time() - t
    print('time:', t, 'seconds')
