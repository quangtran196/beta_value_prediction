# config.py
import os
import json

# Base model configuration with tuned hyperparameters from optimization
BASE_CONFIG = {
    'data_path': 'merged_data_with_error.csv',
    'output_dir': 'beta_model_improved',
    'window_size': 10,
    'prediction_horizon': 10,
    'batch_size': 32,
    'epochs': 100, # <<<<<<<<------ Change to 100 later
    'early_stopping_patience': 15,
    'model_name': 'beta_oscillation_model.h5',
    'random_seed': 42,
    'validation_split': 0.2,
    'test_split': 0.2,
    'sequence_step': 2,  
    'debug_mode': True,

    # Oscillation-specific parameters
    'high_oscillation_threshold': 0.1,      # Threshold for identifying high oscillations
    'high_oscillation_weight': 5,           # Weight multiplier for high oscillation samples
    'augment_ratio': 0.2,                   # Ratio of augmented samples to add
    'oscillation_loss_weights': {           # Weights for custom loss components
        'mse_weight': 1.0,
        'range_weight': 0.3,
        'freq_weight': 0.05
    },

    # Optimized hyperparameters from Bayesian optimization
    'learning_rate': 0.0009256545407983239,  # Tuned learning rate
    'lstm_activation': 'relu',               # Optimized from 'tanh'
    'dense_activation': 'elu',               # Optimized from 'relu'
    'output_activation': 'linear',           # Unchanged
    
    # Optimized architecture parameters
    'lstm_units_1': 64,                      # Confirmed optimal
    'lstm_units_2': 256,                     # Increased from 96
    'dense_units_1': 384,                    # Increased from 128
    'dense_units_2': 96,                     # Increased from 64
    'dropout_rate': 0.15,                    # Optimized from 0.2
    'use_bidirectional': False,              # Confirmed optimal
    'use_residual': True,                    # Enabled (was False)
    'optimizer': 'adam',                     # Confirmed optimal
    'l2_reg': 0.00014222656870530693,       # Reduced from 0.001
    
    # Additional data processing parameters
    'min_group_size': 30,                    # Minimum points needed per parameter group
    'temporal_split': True,                  # Use temporal splitting within groups
    'preserve_temporal_order': True,         # Ensure time ordering in splits
    
    # Step sizes for different data splits
    'train_step_size': 2,                    # Step size for training sequences
    'val_step_size': 5,                      # Step size for validation sequences  
    'test_step_size': 10,                    # Step size for test sequences
    
    # Closed-loop training parameters
    'closed_loop_samples': 2000,             # Number of samples for closed-loop training
    'closed_loop_epochs': 20,         # <<<<<<<<------  Change to 20 la    # Epochs for closed-loop refinement
    'closed_loop_lr_factor': 0.1,            # Learning rate reduction for fine-tuning
}

# Legacy configuration (original hyperparameters before tuning)
LEGACY_CONFIG = {
    'data_path': 'merged_data_with_error.csv',
    'output_dir': 'beta_model_legacy',
    'window_size': 10,
    'prediction_horizon': 10,
    'batch_size': 32,
    'learning_rate': 0.0001,                 # Original learning rate
    'epochs': 100,
    'early_stopping_patience': 15,
    'model_name': 'beta_oscillation_model.h5',
    'random_seed': 42,
    'validation_split': 0.2,
    'test_split': 0.2,
    'sequence_step': 2,  
    'debug_mode': True,

    # Oscillation-specific parameters (legacy defaults)
    'high_oscillation_threshold': 0.1,
    'high_oscillation_weight': 5,
    'augment_ratio': 0.2,
    'oscillation_loss_weights': {
        'mse_weight': 1.0,
        'range_weight': 0.3,
        'freq_weight': 0.05
    },

    'lstm_activation': 'tanh',               # Original activation
    'dense_activation': 'relu',              # Original activation
    'output_activation': 'linear',
    
    # Original architecture parameters
    'lstm_units_1': 64,
    'lstm_units_2': 96,                      # Original size
    'dense_units_1': 128,                    # Original size
    'dense_units_2': 64,                     # Original size
    'dropout_rate': 0.2,                     # Original dropout
    'use_bidirectional': False,
    'use_residual': False,                   # Originally disabled
    'optimizer': 'adam',
    'l2_reg': 0.001,                        # Original regularization
    
    # Data processing parameters
    'min_group_size': 30,
    'temporal_split': True,
    'preserve_temporal_order': True,
    'train_step_size': 2,
    'val_step_size': 5,
    'test_step_size': 10,
    
    # Closed-loop training parameters
    'closed_loop_samples': 2000,
    'closed_loop_epochs': 20,
    'closed_loop_lr_factor': 0.1,
}

# Experimental configuration for testing new ideas
EXPERIMENTAL_CONFIG = {
    **BASE_CONFIG,  # Start with base config
    'output_dir': 'beta_model_experimental',
    
    # Experimental oscillation parameters
    'high_oscillation_threshold': 0.08,      # Lower threshold
    'high_oscillation_weight': 8,            # Higher weight
    'augment_ratio': 0.3,                    # More augmentation
    'oscillation_loss_weights': {
        'mse_weight': 0.7,                   # Less MSE focus
        'range_weight': 0.5,                 # More range focus
        'freq_weight': 0.1                   # More frequency focus
    },
    
    # Progressive training
    'progressive_oscillation_weight': True,   # Enable progressive weighting
    'initial_oscillation_weight': 1.0,        # Start with no extra weight
    'final_oscillation_weight': 10.0,         # End with high weight
    
    # Advanced augmentation
    'use_advanced_augmentation': True,        # Enable advanced augmentation
    'phase_shift_range': (-0.3, 0.3),         # Phase shift range
    'amplitude_range': (0.7, 1.3),            # Amplitude modulation range
    'frequency_range': (0.8, 1.2),            # Frequency perturbation range
}

def get_config(config_path=None, use_legacy=False, use_experimental=False):
    """
    Load configuration with multiple options.
    
    Args:
        config_path: Path to JSON config file (overrides defaults if provided)
                    If None, uses built-in tuned hyperparameters
        use_legacy: If True, use original hyperparameters before tuning
        use_experimental: If True, use experimental configuration
    
    Returns:
        dict: Configuration dictionary
    """
    # Choose base configuration
    if use_experimental:
        config = EXPERIMENTAL_CONFIG.copy()
        print("Using experimental configuration")
    elif use_legacy:
        config = LEGACY_CONFIG.copy()
        print("Using legacy configuration (pre-tuning hyperparameters)")
    else:
        config = BASE_CONFIG.copy()
        print("Using optimized configuration (tuned hyperparameters from config.py)")
    
    # Only load JSON file if explicitly provided AND it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                print(f"JSON config loaded and merged from {config_path}")
        except Exception as e:
            print(f"Error loading JSON config from {config_path}: {e}")
            print("   Continuing with built-in configuration...")
    elif config_path:
        print(f"JSON config file not found: {config_path}")
        print("   Using built-in configuration instead")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    return config

def get_tuned_config():
    """
    Convenience function to get the tuned configuration directly.
    Use this when you want the optimized hyperparameters without any file loading.
    """
    return BASE_CONFIG.copy()

def get_legacy_config():
    """
    Convenience function to get the original configuration.
    Use this for comparison or baseline experiments.
    """
    return LEGACY_CONFIG.copy()

def get_experimental_config():
    """
    Convenience function to get the experimental configuration.
    Use this for testing new ideas and approaches.
    """
    return EXPERIMENTAL_CONFIG.copy()

def save_config_to_file(config, filepath):
    """
    Save a configuration dictionary to a JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path where to save the JSON file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {filepath}")
    except Exception as e:
        print(f"Error saving config to {filepath}: {e}")

def print_config_comparison():
    """Print a comparison between tuned and legacy configurations."""
    tuned = get_tuned_config()
    legacy = get_legacy_config()
    experimental = get_experimental_config()
    
    print("\n=== Configuration Comparison ===")
    print("Key differences between configurations:")
    
    # Find all unique keys
    all_keys = set(tuned.keys()) | set(legacy.keys()) | set(experimental.keys())
    
    differences = []
    for key in sorted(all_keys):
        tuned_val = tuned.get(key, "N/A")
        legacy_val = legacy.get(key, "N/A")
        exp_val = experimental.get(key, "N/A")
        
        # Check if values differ
        if not (tuned_val == legacy_val == exp_val):
            differences.append((key, legacy_val, tuned_val, exp_val))
    
    if differences:
        print(f"\n{'Parameter':<30} {'Legacy':<20} {'Tuned':<20} {'Experimental':<20}")
        print("-" * 90)
        for key, legacy_val, tuned_val, exp_val in differences:
            # Format values for display
            def format_val(val):
                if isinstance(val, float):
                    return f"{val:.6f}"
                elif isinstance(val, dict):
                    return "dict"
                else:
                    return str(val)[:18]
            
            print(f"{key:<30} {format_val(legacy_val):<20} {format_val(tuned_val):<20} {format_val(exp_val):<20}")
    
    return differences

# For backward compatibility, also provide CONFIG (uses tuned hyperparameters by default)
CONFIG = get_tuned_config()

# Example usage patterns:
if __name__ == "__main__":
    print("=== Configuration Examples ===")
    
    # Method 1: Use tuned hyperparameters (recommended)
    config1 = get_config()
    print(f"Method 1 - Tuned config output_dir: {config1['output_dir']}")
    
    # Method 2: Use legacy hyperparameters for comparison
    config2 = get_config(use_legacy=True)
    print(f"Method 2 - Legacy config learning_rate: {config2['learning_rate']}")
    
    # Method 3: Use experimental configuration
    config3 = get_config(use_experimental=True)
    print(f"Method 3 - Experimental config high_oscillation_weight: {config3['high_oscillation_weight']}")
    
    # Method 4: Load from JSON file (if it exists)
    config4 = get_config('custom_config.json')
    print(f"Method 4 - From file config lstm_units_2: {config4['lstm_units_2']}")
    
    # Show comparison
    print_config_comparison()