import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import json
import gc  # Import garbage collector explicitly
import shutil  # For file operations
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Install with: pip install optuna")

from model_builder import build_oscillation_model

class BayesianTuner:
    """Bayesian hyperparameter optimization for beta oscillation models"""
    
    def __init__(self, data, config):
        """Initialize the tuner with data and config"""
        self.data = data
        self.config = config
        self.tuning_dir = os.path.join(config['output_dir'], "tuning_results")
        os.makedirs(self.tuning_dir, exist_ok=True)
        self.best_trial = None
        self.study = None
        
        # Set logging verbosity to suppress warnings
        if OPTUNA_AVAILABLE:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _clear_memory(self):
        """Explicitly clear memory and TensorFlow session"""
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
    
    def _calculate_peak_to_peak_ratio(self, y_pred, y_test):
        """Calculate how well model captures oscillation amplitudes"""
        pp_ratios = []
        
        for i in range(len(y_pred)):
            # Get peak-to-peak for prediction and ground truth
            pred_pp = np.max(y_pred[i]) - np.min(y_pred[i])
            true_pp = np.max(y_test[i]) - np.min(y_test[i])
            
            # Calculate ratio (avoid division by zero)
            if true_pp > 0.01:  # Only consider significant oscillations
                ratio = pred_pp / true_pp
                pp_ratios.append(ratio)
        
        # Average ratio across all sequences
        return np.mean(pp_ratios) if pp_ratios else 0.0
    
    def _create_optimization_plots(self):
        """Create visualization plots for optimization progress"""
        if not OPTUNA_AVAILABLE or not hasattr(self, 'study') or self.study is None:
            print("No study available to create optimization plots")
            return
        
        try:
            # Create plots directory
            plots_dir = os.path.join(self.tuning_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot optimization history
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title('Optimization History')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "optimization_history.png"))
            plt.close()
            
            # Plot parameter importances if we have enough trials
            if len(self.study.trials) > 5:
                try:
                    plt.figure(figsize=(12, 8))
                    optuna.visualization.matplotlib.plot_param_importances(self.study)
                    plt.title('Parameter Importances')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, "param_importances.png"))
                    plt.close()
                except Exception as e:
                    print(f"Could not create parameter importance plot: {e}")
            
            print(f"Optimization plots saved to {plots_dir}")
        except Exception as e:
            print(f"Error creating optimization plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_best_params_to_config(self, best_params=None):
        """Save best parameters to config file"""
        if best_params is None:
            if not hasattr(self, 'best_trial') or self.best_trial is None:
                print("No best trial available to save to config")
                return {}
            best_params = self.best_trial.params
            
        try:
            # Create copy of config with best parameters
            tuned_config = self.config.copy()
            
            # Update with best parameters
            tuned_config.update(best_params)
            
            # Save to file
            tuned_config_path = os.path.join(self.config['output_dir'], "tuned_config.json")
            with open(tuned_config_path, 'w') as f:
                json.dump(tuned_config, f, indent=2)
                
            print(f"Best parameters saved to {tuned_config_path}")
            return tuned_config
        except Exception as e:
            print(f"Error saving best parameters to config: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def objective(self, trial):
        """Objective function for Optuna optimization"""
        # Clear TensorFlow session and memory before each trial
        self._clear_memory()
        
        # Print trial parameters
        print(f"\nTrial {trial.number} parameters:")
        
        try:
            # Define hyperparameters to tune
            params = {
                # Architecture parameters
                'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 256, step=32),
                'lstm_units_2': trial.suggest_int('lstm_units_2', 64, 256, step=32),
                'dense_units_1': trial.suggest_int('dense_units_1', 128, 512, step=64),
                'dense_units_2': trial.suggest_int('dense_units_2', 64, 256, step=32),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
                'use_bidirectional': trial.suggest_categorical('use_bidirectional', [True, False]),
                'use_residual': trial.suggest_categorical('use_residual', [True, False]),
                
                # Training parameters
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop']),
                'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True),
                
                # Activation functions
                'dense_activation': trial.suggest_categorical('dense_activation', 
                                                        ['relu', 'elu']),
                'lstm_activation': trial.suggest_categorical('lstm_activation', 
                                                        ['tanh', 'relu'])
            }
            
            # Print suggested parameters for debugging
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # Create trial config
            trial_config = self.config.copy()
            trial_config.update(params)
            
            # Get input shape and output steps
            input_shape = (self.data['X_train'].shape[1], self.data['X_train'].shape[2])
            output_steps = self.data['y_train'].shape[1]
            
            # Build model with suggested parameters
            model = build_oscillation_model(input_shape, output_steps, trial_config)
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,  # Shorter patience for tuning
                restore_best_weights=True
            )
            
            # Use smaller subset of data for faster tuning
            train_size = min(5000, len(self.data['X_train']))
            val_size = min(1000, len(self.data['X_val']))
            
            X_train_sample = self.data['X_train'][:train_size]
            y_train_sample = self.data['y_train'][:train_size]
            X_val_sample = self.data['X_val'][:val_size]
            y_val_sample = self.data['y_val'][:val_size]
            
            print(f"Starting model training for trial {trial.number}...")
            # Train the model
            history = model.fit(
                X_train_sample, y_train_sample,
                validation_data=(X_val_sample, y_val_sample),
                epochs=20,  # Reduced epochs for faster tuning
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=1  # Show progress for better debugging
            )
            
            # Calculate peak-to-peak performance on validation set
            print(f"Generating predictions for trial {trial.number}...")
            y_pred = model.predict(X_val_sample, verbose=0)
            pp_ratio = self._calculate_peak_to_peak_ratio(y_pred, y_val_sample)
            
            # Get best validation loss
            val_loss = min(history.history['val_loss'])
            
            # Combined score (lower is better)
            # minimize validation loss and maximize pp_ratio towards 1.0
            pp_penalty = abs(1.0 - pp_ratio)  # Penalize deviation from 1.0
            combined_score = val_loss + pp_penalty
            
            # Log results for this trial
            print(f"Trial {trial.number} results:")
            print(f"  val_loss={val_loss:.6f}, pp_ratio={pp_ratio:.2f}, score={combined_score:.6f}")
            
            # Delete model to free memory
            del model
            self._clear_memory()
            
            return combined_score
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            self._clear_memory()
            
            # Return a large value to indicate failure
            return 1e10  # Very large value to indicate this trial failed
    
    def run_optimization(self, n_trials=50, fresh_start=True):
        """Run hyperparameter optimization
        
        Args:
            n_trials: Number of trials to run
            fresh_start: If True, delete existing study and start fresh
        """
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna is not installed. Install with: pip install optuna")
            return {}
            
        print(f"Will run {n_trials} trials of hyperparameter optimization...")
        
        # Storage setup
        storage_path = os.path.join(self.tuning_dir, "optuna_study.db")
        storage = f"sqlite:///{storage_path}"
        study_name = "beta_oscillation_study"
        
        # If fresh start requested, delete existing database
        if fresh_start and os.path.exists(storage_path):
            print(f"Deleting previous study at {storage_path}")
            try:
                # Close any open connections first
                if hasattr(self, 'study') and self.study is not None:
                    del self.study
                
                # Remove file
                os.remove(storage_path)
                print("Previous study deleted successfully.")
            except Exception as e:
                print(f"Warning: Failed to delete previous study: {e}")
                # Try to close SQLite connections
                try:
                    import sqlite3
                    sqlite3.connect(storage_path).close()
                    os.remove(storage_path)
                    print("Retry successful - previous study deleted.")
                except:
                    print("Still cannot delete file. Will try to use a new filename.")
                    storage_path = os.path.join(self.tuning_dir, f"optuna_study_{int(time.time())}.db")
                    storage = f"sqlite:///{storage_path}"
        
        print(f"Starting Bayesian hyperparameter optimization with {n_trials} trials")
        print(f"Using storage: {storage}")
        
        # Create Optuna study
        try:
            self.study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.config['random_seed']),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                storage=storage,
                study_name=study_name,
                load_if_exists=not fresh_start  # Only load existing if not fresh start
            )
            
            # Print existing trials if not fresh start
            if not fresh_start and len(self.study.trials) > 0:
                print(f"Resumed study has {len(self.study.trials)} existing trials")
                print(f"Best value so far: {self.study.best_value:.6f}")
            
            # Optimize with progress tracking
            self.study.optimize(
                self.objective, 
                n_trials=n_trials,
                gc_after_trial=True,  # Enable garbage collection after each trial
                show_progress_bar=True  # Show progress
            )
            
            # Get best parameters
            if hasattr(self.study, 'best_trial'):
                self.best_trial = self.study.best_trial
                best_params = self.best_trial.params
                
                print("\n=== Hyperparameter Optimization Completed ===")
                print(f"Total trials: {len(self.study.trials)}")
                print(f"Best trial: #{self.best_trial.number}")
                print(f"  Value: {self.best_trial.value:.6f}")
                print("  Params:")
                for key, value in best_params.items():
                    print(f"    {key}: {value}")
                
                # Save the study
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                study_path = os.path.join(self.tuning_dir, f"study_final_{timestamp}.pkl")
                try:
                    joblib.dump(self.study, study_path)
                    print(f"Study saved to {study_path}")
                except Exception as e:
                    print(f"Warning: Could not save study to {study_path}: {e}")
                
                # Create and save visualization
                self._create_optimization_plots()
                
                # Update config with best parameters
                self._save_best_params_to_config(best_params)
                
                return best_params
            else:
                print("No best trial found. Optimization may have failed.")
                return {}
        
        except Exception as e:
            print(f"Error during hyperparameter optimization: {e}")
            import traceback
            traceback.print_exc()
            
            # If we have at least one completed trial, save what we have
            if hasattr(self, 'study') and self.study and self.study.trials:
                try:
                    self.best_trial = self.study.best_trial
                    best_params = self.best_trial.params
                    
                    print("\n=== Partial Hyperparameter Optimization Results ===")
                    print(f"Completed {len(self.study.trials)} out of {n_trials} trials")
                    print(f"Best trial so far: #{self.best_trial.number}")
                    print(f"  Value: {self.best_trial.value:.6f}")
                    print("  Params:")
                    for key, value in best_params.items():
                        print(f"    {key}: {value}")
                    
                    # Save partial results
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    study_path = os.path.join(self.tuning_dir, f"study_partial_{timestamp}.pkl")
                    joblib.dump(self.study, study_path)
                    
                    # Update config with best parameters so far
                    self._save_best_params_to_config(best_params)
                    
                    return best_params
                except Exception as inner_e:
                    print(f"Error saving partial results: {inner_e}")
                    return {}
            else:
                print("No completed trials available.")
                return {}
    
    def build_best_model(self):
        """Build model with the best parameters"""
        if self.best_trial is None:
            raise ValueError("No optimization has been run yet. Call run_optimization first.")
        
        # Clear memory before building model
        self._clear_memory()
        
        input_shape = (self.data['X_train'].shape[1], self.data['X_train'].shape[2])
        output_steps = self.data['y_train'].shape[1]
        
        # Create config with best parameters
        best_config = self.config.copy()
        best_config.update(self.best_trial.params)
        
        return build_oscillation_model(input_shape, output_steps, best_config)