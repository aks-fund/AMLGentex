"""
Unit tests for DataPreprocessor class
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.feature_engineering.preprocessor import DataPreprocessor


@pytest.mark.unit
class TestDataPreprocessorInit:
    """Tests for DataPreprocessor initialization"""

    def test_init_basic_config(self, basic_preprocessor_config):
        """Test initialization with basic configuration"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)

        assert preprocessor.num_windows == 2
        assert preprocessor.window_len == 10
        assert preprocessor.train_start_step == 1
        assert preprocessor.train_end_step == 15
        assert preprocessor.bank is None

    def test_init_with_edges(self, preprocessor_config_with_edges):
        """Test initialization with edge features enabled"""
        preprocessor = DataPreprocessor(preprocessor_config_with_edges)

        assert preprocessor.include_edge_features is True


@pytest.mark.unit
class TestLoadData:
    """Tests for load_data method"""

    def test_load_data_basic(self, basic_preprocessor_config, temp_parquet_file):
        """Test basic data loading"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df = preprocessor.load_data(temp_parquet_file)

        # Check that data was loaded
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_filters_source(self, basic_preprocessor_config, sample_transactions_df):
        """Test that source transactions are filtered out"""
        # Add some source transactions
        sample_transactions_df.loc[0:5, 'bankOrig'] = 'source'

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            sample_transactions_df.to_parquet(tmp.name)
            try:
                preprocessor = DataPreprocessor(basic_preprocessor_config)
                df = preprocessor.load_data(tmp.name)

                # Source transactions should be filtered out
                assert 'source' not in df['bankOrig'].values
            finally:
                Path(tmp.name).unlink()

    def test_load_data_drops_columns(self, basic_preprocessor_config, temp_parquet_file):
        """Test that specified columns are dropped"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df = preprocessor.load_data(temp_parquet_file)

        # Balance columns are now kept for balance_at_window_start feature
        dropped_columns = ['type', 'patternID', 'modelType']

        for col in dropped_columns:
            assert col not in df.columns


@pytest.mark.unit
class TestCalNodeFeatures:
    """Tests for cal_node_features method"""

    def test_cal_node_features_basic(self, basic_preprocessor_config, sample_transactions_df):
        """Test basic node feature calculation"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Check that DataFrame is returned
        assert isinstance(df_nodes, pd.DataFrame)
        assert len(df_nodes) > 0

        # Check that essential columns exist
        assert 'account' in df_nodes.columns
        assert 'is_sar' in df_nodes.columns
        assert 'bank' in df_nodes.columns

    def test_cal_node_features_window_features(self, basic_preprocessor_config, sample_transactions_df):
        """Test that window-based features are created"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Check for window-specific feature columns
        window_feature_patterns = ['sum_in', 'mean_in', 'sum_out', 'mean_out',
                                    'sums_spending', 'means_spending']

        for pattern in window_feature_patterns:
            matching_cols = [col for col in df_nodes.columns if pattern in col]
            assert len(matching_cols) > 0, f"No columns found for pattern: {pattern}"

    def test_cal_node_features_no_missing_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no missing values in output"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # No missing values should exist
        assert df_nodes.isnull().sum().sum() == 0

    def test_cal_node_features_no_negative_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no negative values (except bank and timing features that can be negative)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Check no negative values in numeric columns
        # Exclude timing features that can legitimately be negative:
        # - burstiness_* ranges from -1 to 1
        # - time_skew_* can be negative (left-skewed distributions)
        # - volume_trend_* is a correlation coefficient (-1 to 1)
        numeric_cols = df_nodes.select_dtypes(include=[np.number]).columns
        timing_negative_patterns = ['burstiness_', 'time_skew_', 'volume_trend_']
        amount_cols = [col for col in numeric_cols
                       if not any(p in col for p in timing_negative_patterns)]
        assert (df_nodes[amount_cols] >= 0).all().all()

    def test_cal_node_features_are_unique(self, basic_preprocessor_config, sample_transactions_df):
        """Test that all nodes have unique feature vectors"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Get feature columns (exclude account, bank, is_sar which are identifiers/labels)
        feature_cols = [col for col in df_nodes.columns
                       if col not in ['account', 'bank', 'is_sar']]

        # Check that all nodes have unique feature vectors
        n_nodes = len(df_nodes)
        n_unique = len(df_nodes[feature_cols].drop_duplicates())

        assert n_unique == n_nodes, (
            f"Found {n_nodes - n_unique} duplicate feature vectors among {n_nodes} nodes. "
            "Each node should have unique features based on its transaction history."
        )

    def test_cal_node_features_single_window(self, sample_transactions_df):
        """Test with a single window"""
        config = {
            'num_windows': 1,
            'window_len': 15,
            'learning_mode': 'inductive',
            'train_start_step': 1,
            'train_end_step': 15,
            'val_start_step': 16,
            'val_end_step': 22,
            'test_start_step': 23,
            'test_end_step': 30,
            'include_edge_features': False
        }
        preprocessor = DataPreprocessor(config)
        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        assert isinstance(df_nodes, pd.DataFrame)
        assert len(df_nodes) > 0

@pytest.mark.unit
class TestCalEdgeFeatures:
    """Tests for cal_edge_features method"""

    def test_cal_edge_features_basic(self, basic_preprocessor_config, sample_transactions_df):
        """Test basic edge feature calculation"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # Check that DataFrame is returned
        assert isinstance(df_edges, pd.DataFrame)

        # Check that essential columns exist
        assert 'src' in df_edges.columns
        assert 'dst' in df_edges.columns
        assert 'is_sar' in df_edges.columns

    def test_cal_edge_features_window_features(self, basic_preprocessor_config, sample_transactions_df):
        """Test that window-based edge features are created"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # Check for window-specific feature columns
        window_feature_patterns = ['sums_', 'means_', 'medians_', 'stds_', 'maxs_', 'mins_', 'counts_']

        for pattern in window_feature_patterns:
            matching_cols = [col for col in df_edges.columns if pattern in col]
            assert len(matching_cols) > 0, f"No columns found for pattern: {pattern}"

    def test_cal_edge_features_directional(self, basic_preprocessor_config, sample_transactions_df):
        """Test directional vs non-directional edge features"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)

        df_directional = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        df_undirected = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=False
        )

        # Both should return valid DataFrames
        assert isinstance(df_directional, pd.DataFrame)
        assert isinstance(df_undirected, pd.DataFrame)

    def test_cal_edge_features_no_missing_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no missing values in output"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # No missing values should exist
        assert df_edges.isnull().sum().sum() == 0

    def test_cal_edge_features_no_negative_values(self, basic_preprocessor_config, sample_transactions_df):
        """Test that there are no negative values (except src/dst columns)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        df_edges = preprocessor.cal_edge_features(
            sample_transactions_df,
            start_step=1,
            end_step=15,
            directional=True
        )

        # Check no negative values in feature columns (excluding src/dst)
        feature_cols = [col for col in df_edges.columns if col not in ['src', 'dst']]
        numeric_cols = df_edges[feature_cols].select_dtypes(include=[np.number]).columns
        assert (df_edges[numeric_cols] >= 0).all().all()

@pytest.mark.unit
class TestPreprocess:
    """Tests for preprocess method"""

    def test_preprocess_without_edges(self, basic_preprocessor_config, sample_transactions_df):
        """Test preprocessing without edge features (edges are still generated for graph structure)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        result = preprocessor.preprocess(sample_transactions_df)

        # Check that all node sets are returned
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result

        # Edge sets are always generated (for graph structure)
        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result

        # Check that DataFrames are valid
        assert isinstance(result['trainset_nodes'], pd.DataFrame)
        assert isinstance(result['valset_nodes'], pd.DataFrame)
        assert isinstance(result['testset_nodes'], pd.DataFrame)

    def test_preprocess_with_edges(self, preprocessor_config_with_edges, sample_transactions_df):
        """Test preprocessing with edge features"""
        preprocessor = DataPreprocessor(preprocessor_config_with_edges)
        result = preprocessor.preprocess(sample_transactions_df)

        # Check that all node and edge sets are returned
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result
        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result

        # Check that DataFrames are valid
        for key, value in result.items():
            assert isinstance(value, pd.DataFrame), f"{key} is not a DataFrame"


@pytest.mark.unit
class TestCallMethod:
    """Tests for __call__ method"""

    def test_call_method(self, basic_preprocessor_config, temp_parquet_file):
        """Test that __call__ method works end-to-end"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        result = preprocessor(temp_parquet_file)

        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'trainset_nodes' in result
        assert 'valset_nodes' in result
        assert 'testset_nodes' in result

    def test_call_method_with_edges(self, preprocessor_config_with_edges, temp_parquet_file):
        """Test __call__ method with edge features"""
        preprocessor = DataPreprocessor(preprocessor_config_with_edges)
        result = preprocessor(temp_parquet_file)

        # Check that result includes edges
        assert 'trainset_edges' in result
        assert 'valset_edges' in result
        assert 'testset_edges' in result


@pytest.mark.unit
class TestBankFiltering:
    """Tests for bank-specific filtering"""

    def test_bank_filtering_none(self, basic_preprocessor_config, sample_transactions_df):
        """Test when bank is None (no filtering)"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        preprocessor.bank = None

        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Should include accounts from all banks
        assert len(df_nodes) > 0

    def test_bank_filtering_specific_bank(self, basic_preprocessor_config, sample_transactions_df):
        """Test when specific bank is set"""
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        preprocessor.bank = 'BANK001'

        df_nodes = preprocessor.cal_node_features(
            sample_transactions_df,
            start_step=1,
            end_step=15
        )

        # Should only include accounts from BANK001
        if len(df_nodes) > 0:
            assert (df_nodes['bank'] == 'BANK001').all()

    def test_bank_filtering_includes_cross_bank_transactions(self, basic_preprocessor_config):
        """Test that bank filter includes cross-bank transactions in features"""
        # Create test data with cross-bank transactions
        df = pd.DataFrame({
            'step': [1, 1, 1],
            'nameOrig': [100, 100, 200],
            'nameDest': [101, 201, 100],  # 100->101 internal, 100->201 outgoing, 200->100 incoming
            'amount': [1000.0, 2000.0, 3000.0],
            'bankOrig': ['BANK001', 'BANK001', 'BANK002'],
            'bankDest': ['BANK001', 'BANK002', 'BANK001'],
            'daysInBankOrig': [10, 10, 20],
            'daysInBankDest': [15, 25, 10],
            'phoneChangesOrig': [0, 0, 1],
            'phoneChangesDest': [0, 1, 0],
            'isSAR': [0, 0, 0],
        })

        # Process with BANK001 filter
        preprocessor = DataPreprocessor(basic_preprocessor_config)
        preprocessor.bank = 'BANK001'
        df_nodes = preprocessor.cal_node_features(df, start_step=1, end_step=15)

        # Account 100 should have:
        # - sum_out = 1000 + 2000 = 3000 (internal + cross-bank outgoing)
        # - sum_in = 3000 (cross-bank incoming from BANK002)
        account_100 = df_nodes[df_nodes['account'] == 100]
        assert len(account_100) == 1, "Account 100 should exist"

        sum_out_cols = [c for c in account_100.columns if 'sum_out' in c]
        sum_in_cols = [c for c in account_100.columns if 'sum_in' in c]

        total_out = account_100[sum_out_cols].values.sum()
        total_in = account_100[sum_in_cols].values.sum()

        assert total_out == 3000.0, f"Expected sum_out=3000 (internal + cross-bank), got {total_out}"
        assert total_in == 3000.0, f"Expected sum_in=3000 (cross-bank incoming), got {total_in}"


@pytest.mark.unit
class TestSplitByPattern:
    """Tests for split_by_pattern functionality in transductive mode"""

    def test_split_by_pattern_no_overlap(self, transductive_config_with_split_by_pattern,
                                          temp_experiment_with_patterns):
        """Test that split_by_pattern produces non-overlapping pattern splits"""
        preprocessor = DataPreprocessor(transductive_config_with_split_by_pattern)
        result = preprocessor(str(temp_experiment_with_patterns['tx_log_path']))

        df_nodes = result['trainset_nodes']

        # Check masks exist
        assert 'train_mask' in df_nodes.columns
        assert 'val_mask' in df_nodes.columns
        assert 'test_mask' in df_nodes.columns

        # Get SAR accounts in each split
        train_sar = df_nodes[(df_nodes['train_mask']) & (df_nodes['is_sar'] == 1)]['account'].tolist()
        val_sar = df_nodes[(df_nodes['val_mask']) & (df_nodes['is_sar'] == 1)]['account'].tolist()
        test_sar = df_nodes[(df_nodes['test_mask']) & (df_nodes['is_sar'] == 1)]['account'].tolist()

        # No SAR account should appear in multiple splits
        all_sar = train_sar + val_sar + test_sar
        assert len(all_sar) == len(set(all_sar)), "SAR accounts should not appear in multiple splits"

    def test_split_by_pattern_keeps_pattern_together(self, transductive_config_with_split_by_pattern,
                                                      temp_experiment_with_patterns):
        """Test that all accounts from the same pattern are in the same split
        (when accounts don't belong to multiple patterns)"""
        preprocessor = DataPreprocessor(transductive_config_with_split_by_pattern)
        result = preprocessor(str(temp_experiment_with_patterns['tx_log_path']))

        df_nodes = result['trainset_nodes']

        # Load alert_models to get account-to-pattern mapping
        import pandas as pd
        alert_models_df = pd.read_csv(temp_experiment_with_patterns['alert_models_path'])
        account_to_pattern = alert_models_df.groupby('accountID')['modelID'].apply(set).to_dict()

        # Get which split each account is in
        def get_split(row):
            if row['train_mask']:
                return 'train'
            elif row['val_mask']:
                return 'val'
            elif row['test_mask']:
                return 'test'
            return None

        df_nodes['split'] = df_nodes.apply(get_split, axis=1)

        # For each pattern, all its accounts (that are only in this pattern) should be in the same split
        pattern_to_split = {}
        for acc, patterns in account_to_pattern.items():
            if len(patterns) == 1:  # Only check single-pattern accounts
                pattern = next(iter(patterns))
                acc_row = df_nodes[df_nodes['account'] == acc]
                if len(acc_row) > 0:
                    split = acc_row['split'].iloc[0]
                    if pattern in pattern_to_split:
                        assert pattern_to_split[pattern] == split, \
                            f"Pattern {pattern} has accounts in different splits: {pattern_to_split[pattern]} and {split}"
                    else:
                        pattern_to_split[pattern] = split

    def test_split_by_pattern_includes_normal_nodes(self, transductive_config_with_split_by_pattern,
                                                     temp_experiment_with_patterns):
        """Test that normal nodes are also split (randomly) across train/val/test"""
        preprocessor = DataPreprocessor(transductive_config_with_split_by_pattern)
        result = preprocessor(str(temp_experiment_with_patterns['tx_log_path']))

        df_nodes = result['trainset_nodes']

        # Get normal accounts in each split
        train_normal = df_nodes[(df_nodes['train_mask']) & (df_nodes['is_sar'] == 0)]
        val_normal = df_nodes[(df_nodes['val_mask']) & (df_nodes['is_sar'] == 0)]
        test_normal = df_nodes[(df_nodes['test_mask']) & (df_nodes['is_sar'] == 0)]

        # Should have normal nodes in all splits (with the test data we have)
        total_normal = len(train_normal) + len(val_normal) + len(test_normal)
        assert total_normal > 0, "Should have normal nodes"

        # Normal accounts should also not overlap between splits
        train_accs = set(train_normal['account'].tolist())
        val_accs = set(val_normal['account'].tolist())
        test_accs = set(test_normal['account'].tolist())

        assert len(train_accs & val_accs) == 0, "Normal accounts in train and val should not overlap"
        assert len(train_accs & test_accs) == 0, "Normal accounts in train and test should not overlap"
        assert len(val_accs & test_accs) == 0, "Normal accounts in val and test should not overlap"

    def test_split_by_pattern_respects_fractions(self, transductive_config_with_split_by_pattern,
                                                  temp_experiment_with_patterns):
        """Test that the split fractions are approximately respected"""
        preprocessor = DataPreprocessor(transductive_config_with_split_by_pattern)
        result = preprocessor(str(temp_experiment_with_patterns['tx_log_path']))

        df_nodes = result['trainset_nodes']

        n_total = len(df_nodes)
        n_train = df_nodes['train_mask'].sum()
        n_val = df_nodes['val_mask'].sum()
        n_test = df_nodes['test_mask'].sum()

        # All nodes should be in exactly one split
        assert n_train + n_val + n_test == n_total, "All nodes should be in exactly one split"

        # Check fractions are approximately correct (within 20% tolerance due to pattern grouping)
        expected_train = transductive_config_with_split_by_pattern['transductive_train_fraction']
        expected_val = transductive_config_with_split_by_pattern['transductive_val_fraction']
        expected_test = transductive_config_with_split_by_pattern['transductive_test_fraction']

        actual_train_frac = n_train / n_total
        actual_val_frac = n_val / n_total
        actual_test_frac = n_test / n_total

        # Allow 30% tolerance because pattern grouping can skew fractions
        assert abs(actual_train_frac - expected_train) < 0.3, \
            f"Train fraction {actual_train_frac:.2f} differs too much from expected {expected_train}"
        assert abs(actual_val_frac - expected_val) < 0.3, \
            f"Val fraction {actual_val_frac:.2f} differs too much from expected {expected_val}"
        assert abs(actual_test_frac - expected_test) < 0.3, \
            f"Test fraction {actual_test_frac:.2f} differs too much from expected {expected_test}"

    def test_split_by_pattern_deterministic_with_seed(self, transductive_config_with_split_by_pattern,
                                                       temp_experiment_with_patterns):
        """Test that the same seed produces the same split"""
        preprocessor1 = DataPreprocessor(transductive_config_with_split_by_pattern)
        result1 = preprocessor1(str(temp_experiment_with_patterns['tx_log_path']))

        preprocessor2 = DataPreprocessor(transductive_config_with_split_by_pattern)
        result2 = preprocessor2(str(temp_experiment_with_patterns['tx_log_path']))

        df1 = result1['trainset_nodes'].sort_values('account').reset_index(drop=True)
        df2 = result2['trainset_nodes'].sort_values('account').reset_index(drop=True)

        # Masks should be identical
        assert (df1['train_mask'] == df2['train_mask']).all()
        assert (df1['val_mask'] == df2['val_mask']).all()
        assert (df1['test_mask'] == df2['test_mask']).all()

    def test_split_by_pattern_multi_pattern_account(self, transductive_config_with_split_by_pattern,
                                                     sample_transactions_with_patterns_df):
        """Test that accounts in multiple patterns use min pattern ID to determine split"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            temporal_dir = tmpdir / 'temporal'
            spatial_dir = tmpdir / 'spatial'
            temporal_dir.mkdir()
            spatial_dir.mkdir()

            # Save transactions
            tx_log_path = temporal_dir / 'tx_log.parquet'
            sample_transactions_with_patterns_df.to_parquet(tx_log_path)

            # Create alert_models with account 100 in BOTH patterns 1000 and 1001
            alert_models = [
                # Pattern 1000: accounts 100, 101, 102
                {'modelID': 1000, 'type': 'fan_out', 'accountID': 100, 'isMain': True, 'sourceType': 'TRANSFER', 'phase': 0},
                {'modelID': 1000, 'type': 'fan_out', 'accountID': 101, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
                {'modelID': 1000, 'type': 'fan_out', 'accountID': 102, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
                # Pattern 1001: accounts 100, 200, 201, 202 (note: 100 is in both patterns!)
                {'modelID': 1001, 'type': 'fan_in', 'accountID': 100, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
                {'modelID': 1001, 'type': 'fan_in', 'accountID': 200, 'isMain': True, 'sourceType': 'TRANSFER', 'phase': 0},
                {'modelID': 1001, 'type': 'fan_in', 'accountID': 201, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
                {'modelID': 1001, 'type': 'fan_in', 'accountID': 202, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
                # Pattern 1002: accounts 300, 301
                {'modelID': 1002, 'type': 'simple', 'accountID': 300, 'isMain': True, 'sourceType': 'TRANSFER', 'phase': 0},
                {'modelID': 1002, 'type': 'simple', 'accountID': 301, 'isMain': False, 'sourceType': 'TRANSFER', 'phase': 0},
            ]
            alert_models_df = pd.DataFrame(alert_models)
            alert_models_df.to_csv(spatial_dir / 'alert_models.csv', index=False)

            preprocessor = DataPreprocessor(transductive_config_with_split_by_pattern)
            result = preprocessor(str(tx_log_path))

            df_nodes = result['trainset_nodes']

            # Account 100 should be in exactly one split (based on min pattern = 1000)
            acc_100 = df_nodes[df_nodes['account'] == 100]
            assert len(acc_100) == 1, "Account 100 should appear exactly once"

            # Check it's in exactly one split
            masks = acc_100[['train_mask', 'val_mask', 'test_mask']].iloc[0]
            assert masks.sum() == 1, "Account 100 should be in exactly one split"

            # Account 100 should be in the same split as account 101 (both use pattern 1000)
            acc_101 = df_nodes[df_nodes['account'] == 101]
            if len(acc_101) > 0:
                masks_100 = acc_100[['train_mask', 'val_mask', 'test_mask']].iloc[0]
                masks_101 = acc_101[['train_mask', 'val_mask', 'test_mask']].iloc[0]
                # They should be in the same split since they share pattern 1000 (min for account 100)
                assert (masks_100 == masks_101).all(), \
                    "Account 100 and 101 should be in same split (both determined by pattern 1000)"
