"""
Shared pytest fixtures for preprocessing tests
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_nodes_df():
    """Create a sample nodes DataFrame for testing"""
    return pd.DataFrame({
        'account': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'is_sar': [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        'bank': ['BANK001'] * 10,
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })


@pytest.fixture
def sample_edges_df():
    """Create a sample edges DataFrame for testing"""
    return pd.DataFrame({
        'src': [1, 2, 3, 4, 5, 6, 7, 8],
        'dst': [2, 3, 4, 5, 6, 7, 8, 9],
        'amount': np.random.rand(8) * 1000
    })


@pytest.fixture
def sample_alert_members_df():
    """Create a sample alert_members DataFrame for testing"""
    return pd.DataFrame({
        'accountID': [2, 4, 7],
        'reason': ['fan_in', 'fan_out', 'cycle']
    })


@pytest.fixture
def sample_transactions_df():
    """Create a sample transaction DataFrame for testing"""
    np.random.seed(42)
    n_transactions = 100

    # Create realistic transaction data with numeric account IDs
    account_ids = np.random.randint(1000, 1010, n_transactions)
    dest_ids = np.random.randint(1000, 1010, n_transactions)

    # Ensure some transactions go to different banks
    bank_orig = []
    bank_dest = []
    for i in range(n_transactions):
        if i < 70:
            bank_orig.append('BANK001')
            bank_dest.append(np.random.choice(['BANK001', 'sink']))
        elif i < 90:
            bank_orig.append('BANK001')
            bank_dest.append('BANK002')
        else:
            bank_orig.append('BANK002')
            bank_dest.append(np.random.choice(['BANK001', 'BANK002', 'sink']))

    return pd.DataFrame({
        'step': np.random.randint(1, 31, n_transactions),
        'nameOrig': account_ids,
        'nameDest': dest_ids,
        'amount': np.random.uniform(10, 1000, n_transactions),
        'bankOrig': bank_orig,
        'bankDest': bank_dest,
        'type': ['TRANSFER'] * n_transactions,
        'oldbalanceOrig': np.random.uniform(1000, 10000, n_transactions),
        'newbalanceOrig': np.random.uniform(1000, 10000, n_transactions),
        'oldbalanceDest': np.random.uniform(1000, 10000, n_transactions),
        'newbalanceDest': np.random.uniform(1000, 10000, n_transactions),
        'daysInBankOrig': np.random.randint(0, 365, n_transactions),
        'daysInBankDest': np.random.randint(0, 365, n_transactions),
        'phoneChangesOrig': np.random.randint(0, 5, n_transactions),
        'phoneChangesDest': np.random.randint(0, 5, n_transactions),
        'isSAR': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05]),
        'patternID': np.random.randint(-1, 100, n_transactions),  # -1 for generic, 0+ for patterns
        'modelType': np.random.randint(0, 16, n_transactions)  # 0=generic, 1-8=SAR, 10-15=normal
    })


@pytest.fixture
def basic_preprocessor_config():
    """Create a basic configuration for DataPreprocessor (inductive mode)"""
    return {
        'num_windows': 2,
        'window_len': 10,
        'learning_mode': 'inductive',
        'train_start_step': 1,
        'train_end_step': 15,
        'val_start_step': 16,
        'val_end_step': 22,
        'test_start_step': 23,
        'test_end_step': 30,
        'include_edge_features': False
    }


@pytest.fixture
def preprocessor_config_with_edges():
    """Create a configuration with edge features enabled (inductive mode)"""
    return {
        'num_windows': 2,
        'window_len': 10,
        'learning_mode': 'inductive',
        'train_start_step': 1,
        'train_end_step': 15,
        'val_start_step': 16,
        'val_end_step': 22,
        'test_start_step': 23,
        'test_end_step': 30,
        'include_edge_features': True
    }


@pytest.fixture
def temp_parquet_file(sample_transactions_df):
    """Create a temporary parquet file with sample transactions"""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        sample_transactions_df.to_parquet(tmp.name)
        yield tmp.name
        Path(tmp.name).unlink()


@pytest.fixture
def transductive_config_with_split_by_pattern():
    """Create a transductive configuration with split_by_pattern enabled"""
    return {
        'num_windows': 2,
        'window_len': 10,
        'learning_mode': 'transductive',
        'time_start': 1,
        'time_end': 20,
        'transductive_train_fraction': 0.6,
        'transductive_val_fraction': 0.2,
        'transductive_test_fraction': 0.2,
        'split_by_pattern': True,
        'include_edge_features': False,
        'seed': 42
    }


@pytest.fixture
def temp_experiment_with_patterns(sample_transactions_with_patterns_df):
    """
    Create a temporary experiment directory structure with tx_log.parquet and alert_models.csv.

    This allows testing split_by_pattern which requires alert_models.csv.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create experiment directory structure
        temporal_dir = tmpdir / 'temporal'
        spatial_dir = tmpdir / 'spatial'
        temporal_dir.mkdir()
        spatial_dir.mkdir()

        # Save transactions as parquet
        tx_log_path = temporal_dir / 'tx_log.parquet'
        sample_transactions_with_patterns_df.to_parquet(tx_log_path)

        # Create alert_models.csv matching the test data
        # Pattern 1000: accounts 100, 101, 102
        # Pattern 1001: accounts 200, 201, 202
        # Pattern 1002: accounts 300, 301
        alert_models = [
            {'modelID': 1000, 'type': 'fan_out', 'accountID': 100, 'isMain': True, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1000, 'type': 'fan_out', 'accountID': 101, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1000, 'type': 'fan_out', 'accountID': 102, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1001, 'type': 'fan_in', 'accountID': 200, 'isMain': True, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1001, 'type': 'fan_in', 'accountID': 201, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1001, 'type': 'fan_in', 'accountID': 202, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1002, 'type': 'simple', 'accountID': 300, 'isMain': True, 'sourceType': 'TRANSFER', 'phase': 0},
            {'modelID': 1002, 'type': 'simple', 'accountID': 301, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
        ]
        alert_models_df = pd.DataFrame(alert_models)
        alert_models_path = spatial_dir / 'alert_models.csv'
        alert_models_df.to_csv(alert_models_path, index=False)

        yield {
            'experiment_dir': tmpdir,
            'tx_log_path': tx_log_path,
            'alert_models_path': alert_models_path,
        }


@pytest.fixture
def sample_transactions_with_patterns_df():
    """
    Create a sample transaction DataFrame with controlled SAR patterns.

    Creates 3 distinct SAR patterns (1000, 1001, 1002) with multiple accounts each,
    plus normal transactions with patternID=-1 or normal model IDs (0-9).
    """
    np.random.seed(42)

    transactions = []

    # SAR Pattern 1000: accounts 100, 101, 102 (fan-out pattern)
    # Account 100 sends to 101 and 102
    for step in range(1, 6):
        transactions.append({
            'step': step, 'nameOrig': 100, 'nameDest': 101, 'amount': 500.0,
            'bankOrig': 'BANK001', 'bankDest': 'BANK001',
            'daysInBankOrig': 30, 'daysInBankDest': 30,
            'phoneChangesOrig': 0, 'phoneChangesDest': 0,
            'isSAR': 1, 'patternID': 1000, 'modelType': 1,
            'oldbalanceOrig': 5000, 'newbalanceOrig': 4500,
            'oldbalanceDest': 1000, 'newbalanceDest': 1500,
        })
        transactions.append({
            'step': step, 'nameOrig': 100, 'nameDest': 102, 'amount': 500.0,
            'bankOrig': 'BANK001', 'bankDest': 'BANK001',
            'daysInBankOrig': 30, 'daysInBankDest': 30,
            'phoneChangesOrig': 0, 'phoneChangesDest': 0,
            'isSAR': 1, 'patternID': 1000, 'modelType': 1,
            'oldbalanceOrig': 4500, 'newbalanceOrig': 4000,
            'oldbalanceDest': 1000, 'newbalanceDest': 1500,
        })

    # SAR Pattern 1001: accounts 200, 201, 202 (fan-in pattern)
    # Accounts 201 and 202 send to 200
    for step in range(1, 6):
        transactions.append({
            'step': step, 'nameOrig': 201, 'nameDest': 200, 'amount': 300.0,
            'bankOrig': 'BANK001', 'bankDest': 'BANK001',
            'daysInBankOrig': 30, 'daysInBankDest': 30,
            'phoneChangesOrig': 0, 'phoneChangesDest': 0,
            'isSAR': 1, 'patternID': 1001, 'modelType': 2,
            'oldbalanceOrig': 3000, 'newbalanceOrig': 2700,
            'oldbalanceDest': 1000, 'newbalanceDest': 1300,
        })
        transactions.append({
            'step': step, 'nameOrig': 202, 'nameDest': 200, 'amount': 300.0,
            'bankOrig': 'BANK001', 'bankDest': 'BANK001',
            'daysInBankOrig': 30, 'daysInBankDest': 30,
            'phoneChangesOrig': 0, 'phoneChangesDest': 0,
            'isSAR': 1, 'patternID': 1001, 'modelType': 2,
            'oldbalanceOrig': 3000, 'newbalanceOrig': 2700,
            'oldbalanceDest': 1300, 'newbalanceDest': 1600,
        })

    # SAR Pattern 1002: accounts 300, 301 (simple transfer)
    for step in range(1, 6):
        transactions.append({
            'step': step, 'nameOrig': 300, 'nameDest': 301, 'amount': 1000.0,
            'bankOrig': 'BANK002', 'bankDest': 'BANK002',
            'daysInBankOrig': 30, 'daysInBankDest': 30,
            'phoneChangesOrig': 0, 'phoneChangesDest': 0,
            'isSAR': 1, 'patternID': 1002, 'modelType': 3,
            'oldbalanceOrig': 10000, 'newbalanceOrig': 9000,
            'oldbalanceDest': 2000, 'newbalanceDest': 3000,
        })

    # Normal transactions (patternID = -1 or normal model IDs 0-9)
    normal_accounts = [400, 401, 402, 403, 404, 500, 501, 502, 503, 504]
    for i, acc in enumerate(normal_accounts):
        for step in range(1, 11):
            dest = normal_accounts[(i + 1) % len(normal_accounts)]
            transactions.append({
                'step': step, 'nameOrig': acc, 'nameDest': dest,
                'amount': np.random.uniform(100, 500),
                'bankOrig': 'BANK001' if acc < 500 else 'BANK002',
                'bankDest': 'BANK001' if dest < 500 else 'BANK002',
                'daysInBankOrig': 30, 'daysInBankDest': 30,
                'phoneChangesOrig': 0, 'phoneChangesDest': 0,
                'isSAR': 0, 'patternID': i % 10,  # Normal model IDs 0-9
                'modelType': 10 + (i % 6),  # Normal model types 10-15
                'oldbalanceOrig': 5000, 'newbalanceOrig': 4500,
                'oldbalanceDest': 2000, 'newbalanceDest': 2500,
            })

    # Also add some normal transactions for SAR accounts (they have normal activity too)
    sar_accounts = [100, 101, 102, 200, 201, 202, 300, 301]
    for acc in sar_accounts:
        for step in range(6, 11):  # Different time range
            dest = 999  # Sink
            transactions.append({
                'step': step, 'nameOrig': acc, 'nameDest': dest,
                'amount': np.random.uniform(50, 200),
                'bankOrig': 'BANK001' if acc < 300 else 'BANK002',
                'bankDest': 'sink',
                'daysInBankOrig': 30, 'daysInBankDest': 0,
                'phoneChangesOrig': 0, 'phoneChangesDest': 0,
                'isSAR': 0, 'patternID': -1,  # Normal transaction
                'modelType': 0,
                'oldbalanceOrig': 5000, 'newbalanceOrig': 4800,
                'oldbalanceDest': 0, 'newbalanceDest': 200,
            })

    df = pd.DataFrame(transactions)
    df['type'] = 'TRANSFER'
    return df
