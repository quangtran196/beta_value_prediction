
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import modules 
from config import get_config, save_config_to_file
from data_processor import DataProcessor
from model_builder import build_oscillation_model
from trainer import ModelTrainer, OscillationAwareLoss
from evaluator import ModelEvaluator

def check_gpu():
    """Check for GPU availability"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available: {len(gpus)} device(s)")
            return True
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
            return False
    else:
        print("No GPU found, using CPU")
        return False

def train_model(config_path=None, use_legacy=False, use_tuning=False):
    """Train a new model with improved training pipeline"""
    print("Starting improved model training pipeline...")
    
    # Load config
    config = get_config(config_path, use_legacy=use_legacy)
    
    # Add oscillation-specific parameters if not present
    if 'high_oscillation_threshold' not in config:
        config['high_oscillation_threshold'] = 0.1
    if 'high_oscillation_weight' not in config:
        config['high_oscillation_weight'] = 5
    
    # Set random seeds
    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Step 1: Prepare data
    print("\n=== Data Preparation ===")
    data_processor = DataProcessor(config)
    data = data_processor.prepare_data()
    
    # Optional: Run hyperparameter tuning
    if use_tuning:
        print("\n=== Hyperparameter Tuning ===")
        try:
            from hyperparameter_tuner import BayesianTuner
            tuner = BayesianTuner(data, config)
            best_params = tuner.run_optimization(n_trials=30, fresh_start=True)
            
            # Update config with best parameters
            config.update(best_params)
            
            # Save tuned config
            tuned_config_path = os.path.join(config['output_dir'], "tuned_config.json")
            save_config_to_file(config, tuned_config_path)
            print(f"Tuned configuration saved to: {tuned_config_path}")
        except ImportError:
            print("Warning: hyperparameter_tuner not available. Skipping tuning.")
        except Exception as e:
            print(f"Warning: Tuning failed with error: {e}. Continuing with default parameters.")
    
    # Step 2: Build model
    print("\n=== Model Building ===")
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    output_steps = data['y_train'].shape[1]
    model = build_oscillation_model(input_shape, output_steps, config)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Step 3: Train model with improved pipeline
    print("\n=== Model Training (Improved Pipeline) ===")
    trainer = ModelTrainer(model, data, config)
    trained_model = trainer.run_training_pipeline()
    
    # Step 4: Evaluate model and generate predictions
    print("\n=== Model Evaluation ===")
    evaluator = ModelEvaluator(trained_model, data, config)
    metrics = evaluator.run_evaluation()
    
    # Step 5: Save training configuration and results
    results_summary = {
        'config': config,
        'metrics': metrics,
        'training_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu_available': has_gpu,
        'data_stats': {
            'train_samples': len(data['X_train']),
            'val_samples': len(data['X_val']),
            'test_samples': len(data['X_test'])
        }
    }
    
    summary_path = os.path.join(config['output_dir'], "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n=== Training Pipeline Complete ===")
    print(f"  Final metrics:")
    print(f"  Open-loop MSE: {metrics['mse_open']:.6f}")
    print(f"  Closed-loop MSE: {metrics['mse_closed']:.6f}")
    print(f"  Improvement: {metrics['improvement']:.1f}%")
    
    # Path to saved predictions
    pkl_path = os.path.join(config['output_dir'], "evaluation", "all_predictions.pkl")
    print(f"\n  Model saved to: {os.path.join(config['output_dir'], config['model_name'])}")
    print(f"  Predictions saved to: {pkl_path}")
    print(f"  Training summary saved to: {summary_path}")
    print(f"\n  To visualize predictions, run:")
    print(f"   python prediction_plotter.py --predictions_file {pkl_path} --output_path plots/results.png")
    
    return trained_model, data, metrics

def evaluate_model(model_path, config_path=None, use_legacy=False):
    """Evaluate an existing model"""
    print(f"Evaluating existing model: {model_path}")
    
    # Load config
    config = get_config(config_path, use_legacy=use_legacy)
    
    # Try to load model with custom loss
    try:
        # Register custom loss
        custom_objects = {
            'OscillationAwareLoss': OscillationAwareLoss,
            'oscillation_aware_loss': OscillationAwareLoss()
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded with custom loss function")
    except:
        # Fallback to standard loading
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')
        print("Model loaded with standard MSE loss")
    
    # Prepare data
    data_processor = DataProcessor(config)
    data = data_processor.prepare_data()
    
    # Evaluate model
    evaluator = ModelEvaluator(model, data, config)
    metrics = evaluator.run_evaluation()
    
    print("\n=== Evaluation Complete ===")
    print(f"  Metrics:")
    print(f"  Open-loop MSE: {metrics['mse_open']:.6f}")
    print(f"  Closed-loop MSE: {metrics['mse_closed']:.6f}")
    print(f"  Improvement: {metrics['improvement']:.1f}%")
    
    pkl_path = os.path.join(config['output_dir'], "evaluation", "all_predictions.pkl")
    print(f"\n  Predictions saved to: {pkl_path}")
    
    return model, data, metrics

def compare_models(model1_path, model2_path, config_path=None):
    """Compare two models side by side"""
    print("Comparing two models...")
    
    # Load config
    config = get_config(config_path)
    
    # Prepare data once
    data_processor = DataProcessor(config)
    data = data_processor.prepare_data()
    
    # Evaluate both models
    print(f"\n=== Evaluating Model 1: {model1_path} ===")
    config1 = config.copy()
    config1['output_dir'] = os.path.join(config['output_dir'], "model1_eval")
    os.makedirs(config1['output_dir'], exist_ok=True)
    
    try:
        model1 = load_model(model1_path, custom_objects={'OscillationAwareLoss': OscillationAwareLoss})
    except:
        model1 = load_model(model1_path, compile=False)
        model1.compile(optimizer='adam', loss='mse')
    
    evaluator1 = ModelEvaluator(model1, data, config1)
    metrics1 = evaluator1.run_evaluation()
    
    print(f"\n=== Evaluating Model 2: {model2_path} ===")
    config2 = config.copy()
    config2['output_dir'] = os.path.join(config['output_dir'], "model2_eval")
    os.makedirs(config2['output_dir'], exist_ok=True)
    
    try:
        model2 = load_model(model2_path, custom_objects={'OscillationAwareLoss': OscillationAwareLoss})
    except:
        model2 = load_model(model2_path, compile=False)
        model2.compile(optimizer='adam', loss='mse')
    
    evaluator2 = ModelEvaluator(model2, data, config2)
    metrics2 = evaluator2.run_evaluation()
    
    # Print comparison
    print("\n=== Model Comparison ===")
    print(f"{'Metric':<20} {'Model 1':<15} {'Model 2':<15} {'Improvement':<15}")
    print("-" * 65)
    
    for metric in ['mse_open', 'mse_closed', 'mae_open', 'mae_closed']:
        val1 = metrics1[metric]
        val2 = metrics2[metric]
        improvement = 100 * (val1 - val2) / val1 if val1 > 0 else 0
        print(f"{metric:<20} {val1:<15.6f} {val2:<15.6f} {improvement:<15.2f}%")
    
    # Save comparison
    comparison_path = os.path.join(config['output_dir'], "model_comparison.json")
    comparison_data = {
        'model1': {
            'path': model1_path,
            'metrics': metrics1
        },
        'model2': {
            'path': model2_path,
            'metrics': metrics2
        },
        'comparison_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n Comparison saved to: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='Beta Oscillation Prediction Model (Improved)')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'compare', 'tune'],
                        help='Mode: train, evaluate, compare, or tune')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to configuration JSON file (optional)')
    parser.add_argument('--use_legacy', action='store_true',
                        help='Use legacy hyperparameters instead of optimized ones')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model file (required for evaluate mode)')
    parser.add_argument('--model2_path', type=str, default=None,
                        help='Path to second model file (required for compare mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--use_tuning', action='store_true',
                        help='Run hyperparameter tuning before training')
    
    args = parser.parse_args()
    
    # Get configuration
    print("=== Loading Configuration ===")
    config = get_config(args.config_path, use_legacy=args.use_legacy)
    
    # Override output_dir if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
        os.makedirs(config['output_dir'], exist_ok=True)
        print(f"Output directory: {config['output_dir']}")
        
        # Save config for reference
        config_backup_path = os.path.join(config['output_dir'], "used_config.json")
        save_config_to_file(config, config_backup_path)
    
    # Display configuration summary
    print(f"\n=== Configuration Summary ===")
    print(f"   Model Architecture:")
    print(f"   LSTM Units: {config['lstm_units_1']} → {config['lstm_units_2']}")
    print(f"   Dense Units: {config['dense_units_1']} → {config['dense_units_2']}")
    print(f"   Activation: LSTM={config.get('lstm_activation', 'tanh')}, Dense={config.get('dense_activation', 'relu')}")
    print(f"   Training Parameters:")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   High Oscillation Weight: {config.get('high_oscillation_weight', 5)}")
    
    # Run in specified mode
    start_time = time.time()
    
    if args.mode == 'train':
        print("\n===== TRAINING MODEL (IMPROVED PIPELINE) =====")
        train_model(args.config_path, args.use_legacy, args.use_tuning)
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("  Error: model_path is required for evaluate mode")
            return
        evaluate_model(args.model_path, args.config_path, args.use_legacy)
    
    elif args.mode == 'compare':
        if args.model_path is None or args.model2_path is None:
            print("  Error: both model_path and model2_path are required for compare mode")
            return
        compare_models(args.model_path, args.model2_path, args.config_path)
    
    elif args.mode == 'tune':
        print("\n===== HYPERPARAMETER TUNING =====")
        try:
            from hyperparameter_tuner import BayesianTuner
            # Prepare data
            data_processor = DataProcessor(config)
            data = data_processor.prepare_data()
            
            # Run tuning
            tuner = BayesianTuner(data, config)
            best_params = tuner.run_optimization(n_trials=50, fresh_start=True)
            
            print("\n  Tuning complete! Best parameters saved.")
            print("To train with tuned parameters, run:")
            print(f"   python main.py --mode train --config_path {os.path.join(config['output_dir'], 'tuning_results/tuned_config.json')}")
        except ImportError:
            print("Error: hyperparameter_tuner module not found.")
            print("Make sure hyperparameter_tuner.py is in the same directory.")
        except Exception as e:
            print(f"Error during tuning: {e}")
            import traceback
            traceback.print_exc()
    
    end_time = time.time()
    print(f"\n    Total execution time: {(end_time - start_time)/60:.2f} minutes")
    print("  Execution completed successfully!")

if __name__ == "__main__":
    main()

    """
    Example usage commands:
    
    # Train with improved pipeline and optimized hyperparameters
    python main.py --mode train
    
    # Train with hyperparameter tuning first
    python main.py --mode train --use_tuning
    
    # Train with legacy hyperparameters
    python main.py --mode train --use_legacy
    
    # Train with custom output directory
    python main.py --mode train --output_dir experiments/improved_model
    
    # Evaluate existing model
    python main.py --mode evaluate --model_path beta_model_improved/beta_oscillation_model.h5
    
    # Compare two models
    python main.py --mode compare --model_path model1.h5 --model2_path model2.h5
    
    # Run hyperparameter tuning only
    python main.py --mode tune --output_dir tuning_experiments
    
    # After training, visualize predictions:
    python prediction_plotter.py --predictions_file beta_model_improved/evaluation/all_predictions.pkl --output_path plots/results.png
    """