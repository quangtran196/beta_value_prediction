import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from config import CONFIG

class DataProcessor:
    def __init__(self, config=CONFIG):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load raw data without scaling"""
        df = pd.read_csv(self.config['data_path'])
        
        if self.config['debug_mode']:
            print(f"Raw data shape: {df.shape}")
            print(f"Beta value range: [{df['beta_value'].min()}, {df['beta_value'].max()}]")
            print(f"Columns: {list(df.columns)}")
        
        return df
    
    def split_data_by_time(self, df):
        """Split data by time BEFORE any processing to prevent leakage"""
        print("Splitting data by time within each parameter group...")
        
        # Get unique amplitude-frequency combinations
        unique_groups = df.groupby(['amplitude', 'frequency']).groups.keys()
        unique_groups = list(unique_groups)
        
        print(f"Found {len(unique_groups)} unique parameter combinations")
        
        # Create lists for each split
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for amp, freq in unique_groups:
            group_data = df[(df['amplitude'] == amp) & (df['frequency'] == freq)]
            
            # Skip if group is too small
            if len(group_data) < self.config['min_group_size']:
                print(f"Warning: Skipping small group (amp={amp:.1f}, freq={freq:.1f}) with only {len(group_data)} points")
                continue
            
            # Sort by time to maintain temporal order
            group_data = group_data.sort_values('time').reset_index(drop=True)
            
            # Calculate split indices
            n = len(group_data)
            train_end = int(n * (1 - self.config['validation_split'] - self.config['test_split']))
            val_end = int(n * (1 - self.config['test_split']))
            
            # Split this group temporally
            train_group = group_data.iloc[:train_end]
            val_group = group_data.iloc[train_end:val_end]
            test_group = group_data.iloc[val_end:]
            
            # Add to respective lists
            if len(train_group) > 0:
                train_dfs.append(train_group)
            if len(val_group) > 0:
                val_dfs.append(val_group)
            if len(test_group) > 0:
                test_dfs.append(test_group)
        
        # Combine all groups for each split
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def scale_data(self, train_df, val_df, test_df):
        """Scale data using only training set statistics"""
        print("Scaling data using training set statistics...")
        
        # Fit scaler on training data only
        features_to_scale = ['amplitude', 'frequency', 'beta_value']
        self.scaler.fit(train_df[features_to_scale])
        
        # Transform all splits
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()
        
        train_scaled[features_to_scale] = self.scaler.transform(train_df[features_to_scale])
        val_scaled[features_to_scale] = self.scaler.transform(val_df[features_to_scale])
        test_scaled[features_to_scale] = self.scaler.transform(test_df[features_to_scale])
        
        return train_scaled, val_scaled, test_scaled
    
    def create_sequences_from_df(self, df, step_size=2, split_type="train"):
        """Create sequences from a dataframe"""
        window_size = self.config['window_size']
        horizon = self.config['prediction_horizon']
        
        print(f"Creating {split_type} sequences (step: {step_size}, window: {window_size}, horizon: {horizon})")
        
        if len(df) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Group by amplitude and frequency
        groups = df.groupby(['amplitude', 'frequency'])
        
        X, y = [], []
        group_info = []
        
        for (amp, freq), group in groups:
            if len(group) <= window_size + horizon:
                continue
                
            # Sort by time
            group = group.sort_values('time')
            group_data = group[['amplitude', 'frequency', 'beta_value']].to_numpy()
            
            # Create sequences with specified step size
            for i in range(0, len(group_data) - window_size - horizon, step_size):
                # Input sequence
                X.append(group_data[i:i + window_size])
                # Target sequence
                y.append(group_data[i + window_size:i + window_size + horizon, 2])  # beta values only
                # Store group info
                group_info.append((amp, freq))
                
                # For training, apply oversampling for high oscillation sequences
                if split_type == "train":
                    beta_values = group_data[i:i + window_size, 2]
                    beta_range = np.max(beta_values) - np.min(beta_values)
                    
                    if beta_range > self.config['high_oscillation_threshold']:
                        for _ in range(self.config['high_oscillation_weight'] - 1):
                            X.append(group_data[i:i + window_size])
                            y.append(group_data[i + window_size:i + window_size + horizon, 2])
                            group_info.append((amp, freq))
        
        X = np.array(X) if X else np.array([]).reshape(0, window_size, 3)
        y = np.array(y) if y else np.array([]).reshape(0, horizon)
        group_info = np.array(group_info) if group_info else np.array([]).reshape(0, 2)
        
        if self.config['debug_mode']:
            print(f"Created {len(X)} sequences for {split_type}")
            
        return X, y, group_info
    
    def prepare_data(self):
        """Complete data preparation pipeline with proper splitting"""
        # Step 1: Load raw data
        print("=== Step 1: Loading raw data ===")
        raw_df = self.load_data()
        
        # Step 2: Split BEFORE scaling to prevent leakage
        print("\n=== Step 2: Splitting data temporally ===")
        train_df, val_df, test_df = self.split_data_by_time(raw_df)
        
        # Step 3: Scale data using only training statistics
        print("\n=== Step 3: Scaling data ===")
        train_scaled, val_scaled, test_scaled = self.scale_data(train_df, val_df, test_df)
        
        # Step 4: Create sequences
        print("\n=== Step 4: Creating sequences ===")
        
        # Use different step sizes for each split
        X_train, y_train, groups_train = self.create_sequences_from_df(
            train_scaled, step_size=self.config['train_step_size'], split_type="train"
        )
        
        X_val, y_val, groups_val = self.create_sequences_from_df(
            val_scaled, step_size=self.config['val_step_size'], split_type="val"
        )
        
        X_test, y_test, groups_test = self.create_sequences_from_df(
            test_scaled, step_size=self.config['test_step_size'], split_type="test"
        )
        
        # Step 5: Summary
        print("\n=== Step 5: Data preparation summary ===")
        print(f"Training:   {X_train.shape[0]} sequences")
        print(f"Validation: {X_val.shape[0]} sequences")  
        print(f"Test:       {X_test.shape[0]} sequences")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'groups_train': groups_train,
            'groups_val': groups_val,
            'groups_test': groups_test,
            'scaler': self.scaler,
            'original_df': raw_df
        }