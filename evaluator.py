import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

from config import CONFIG
from data_processor import DataProcessor

class ModelEvaluator:
    def __init__(self, model, data, config=CONFIG):
        self.model = model
        self.data = data
        self.config = config
        self.eval_dir = os.path.join(config['output_dir'], "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
    
    def calculate_oscillation_metrics(self, y_true, y_pred):
        """Calculate oscillation-specific metrics"""
        metrics = {
            'peak_to_peak_ratios': [],
            'frequency_errors': [],
            'phase_errors': [],
            'oscillation_mse_by_range': {'low': [], 'medium': [], 'high': []}
        }
        
        # Get oscillation threshold from config or use default
        threshold = self.config.get('high_oscillation_threshold', 0.1)
        
        for i in range(len(y_true)):
            # Peak-to-peak analysis
            true_pp = np.max(y_true[i]) - np.min(y_true[i])
            pred_pp = np.max(y_pred[i]) - np.min(y_pred[i])
            
            if true_pp > 0.01:  # Only for significant oscillations
                pp_ratio = pred_pp / true_pp
                metrics['peak_to_peak_ratios'].append(pp_ratio)
                
                # Categorize by oscillation magnitude
                if true_pp < threshold * 0.5:
                    category = 'low'
                elif true_pp < threshold * 1.5:
                    category = 'medium'
                else:
                    category = 'high'
                
                mse = np.mean((y_true[i] - y_pred[i])**2)
                metrics['oscillation_mse_by_range'][category].append(mse)
            
            # Simple frequency analysis using zero-crossings
            true_mean = np.mean(y_true[i])
            pred_mean = np.mean(y_pred[i])
            
            true_crossings = np.sum(np.diff(np.sign(y_true[i] - true_mean)) != 0)
            pred_crossings = np.sum(np.diff(np.sign(y_pred[i] - pred_mean)) != 0)
            
            if true_crossings > 0:
                freq_error = abs(pred_crossings - true_crossings) / true_crossings
                metrics['frequency_errors'].append(freq_error)
        
        # Calculate summary statistics
        summary = {}
        
        if metrics['peak_to_peak_ratios']:
            pp_ratios = np.array(metrics['peak_to_peak_ratios'])
            summary['pp_ratio_mean'] = float(np.mean(pp_ratios))
            summary['pp_ratio_std'] = float(np.std(pp_ratios))
            summary['pp_ratio_median'] = float(np.median(pp_ratios))
            # Percentage within 20% of true value
            summary['pp_ratio_within_20pct'] = float(np.sum((pp_ratios > 0.8) & (pp_ratios < 1.2)) / len(pp_ratios) * 100)
        
        if metrics['frequency_errors']:
            summary['freq_error_mean'] = float(np.mean(metrics['frequency_errors']))
            summary['freq_error_std'] = float(np.std(metrics['frequency_errors']))
        
        # MSE by oscillation range
        for range_type in ['low', 'medium', 'high']:
            if metrics['oscillation_mse_by_range'][range_type]:
                summary[f'mse_{range_type}_osc'] = float(np.mean(metrics['oscillation_mse_by_range'][range_type]))
                summary[f'n_samples_{range_type}_osc'] = len(metrics['oscillation_mse_by_range'][range_type])
        
        return metrics, summary
    
    def generate_closed_loop_predictions(self):
        """Generate closed-loop predictions for the entire test set"""
        print("Generating closed-loop predictions for test set...")
        
        # Get test data
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        groups_test = self.data['groups_test']
        scaler = self.data['scaler']  # Get the scaler
        
        # Generate open-loop predictions first
        print("Generating open-loop predictions...")
        y_pred_open = self.model.predict(X_test, verbose=1)
        
        # Group test data by amplitude-frequency
        groups = {}
        for i in range(len(X_test)):
            key = tuple(groups_test[i])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        print(f"Found {len(groups)} unique parameter groups in test set")
        
        # Generate closed-loop predictions
        print("Generating closed-loop predictions...")
        y_pred_closed = np.zeros_like(y_pred_open)
        
        for (amp_scaled, freq_scaled), indices in groups.items():
            if len(indices) == 0:
                continue
            
            # Unscale amplitude and frequency for display
            # Create dummy array for inverse transform
            dummy = np.array([[amp_scaled, freq_scaled, 0.0]])  # beta value doesn't matter here
            unscaled = scaler.inverse_transform(dummy)
            amp_unscaled = unscaled[0, 0]
            freq_unscaled = unscaled[0, 1]
            
            print(f"Processing group: amp={amp_unscaled:.2f} mA, freq={freq_unscaled:.1f} Hz ({len(indices)} sequences)")
            
            # Start with the first sequence in this group
            current_sequence = X_test[indices[0]].copy()
            
            # Generate predictions for all sequences in this group
            for seq_idx, test_idx in enumerate(indices):
                # Predict next horizon steps
                pred = self.model.predict(np.expand_dims(current_sequence, axis=0), verbose=0)[0]
                y_pred_closed[test_idx] = pred
                
                # Update sequence for next prediction
                if seq_idx < len(indices) - 1:
                    # Shift the window forward by prediction horizon
                    new_sequence = np.zeros_like(current_sequence)
                    
                    # Keep amplitude and frequency constant (in scaled form)
                    new_sequence[:, 0] = amp_scaled
                    new_sequence[:, 1] = freq_scaled
                    
                    # Shift beta values
                    shift_amount = self.config['prediction_horizon']
                    if shift_amount < self.config['window_size']:
                        # Shift existing values
                        new_sequence[:-shift_amount, 2] = current_sequence[shift_amount:, 2]
                        # Add new predictions
                        new_sequence[-shift_amount:, 2] = pred[:shift_amount]
                    else:
                        # Use only predictions
                        new_sequence[:, 2] = pred[:self.config['window_size']]
                    
                    current_sequence = new_sequence
        
        # Calculate standard metrics
        print("\nCalculating standard metrics...")
        all_true = []
        all_open_loop = []
        all_closed_loop = []
        
        for i in range(len(y_test)):
            all_true.extend(y_test[i])
            all_open_loop.extend(y_pred_open[i])
            all_closed_loop.extend(y_pred_closed[i])
        
        # Convert to numpy arrays
        all_true = np.array(all_true)
        all_open_loop = np.array(all_open_loop)
        all_closed_loop = np.array(all_closed_loop)
        
        # Calculate metrics on SCALED data (as the model works with scaled data)
        mse_open = np.mean((all_open_loop - all_true)**2)
        mse_closed = np.mean((all_closed_loop - all_true)**2)
        mae_open = np.mean(np.abs(all_open_loop - all_true))
        mae_closed = np.mean(np.abs(all_closed_loop - all_true))
        
        # Also calculate metrics on UNSCALED data for interpretability
        print("\nCalculating metrics on unscaled data...")
        
        # Create dummy arrays for inverse transform
        n_points = len(all_true)
        # need to know which amp/freq each point belongs to
        # use average values for demonstration
        dummy_true = np.column_stack([
            np.zeros(n_points),  # dummy amplitude
            np.zeros(n_points),  # dummy frequency
            all_true             # beta values
        ])
        dummy_open = dummy_true.copy()
        dummy_open[:, 2] = all_open_loop
        dummy_closed = dummy_true.copy()
        dummy_closed[:, 2] = all_closed_loop
        
        unscaled_true = scaler.inverse_transform(dummy_true)[:, 2]
        unscaled_open = scaler.inverse_transform(dummy_open)[:, 2]
        unscaled_closed = scaler.inverse_transform(dummy_closed)[:, 2]
        
        mse_open_unscaled = np.mean((unscaled_open - unscaled_true)**2)
        mse_closed_unscaled = np.mean((unscaled_closed - unscaled_true)**2)
        
        print(f"\nStandard Metrics (Scaled Data):")
        print(f"Open-loop MSE: {mse_open:.6f}, MAE: {mae_open:.6f}")
        print(f"Closed-loop MSE: {mse_closed:.6f}, MAE: {mae_closed:.6f}")
        
        print(f"\nStandard Metrics (Unscaled Data):")
        print(f"Open-loop MSE: {mse_open_unscaled:.6f}")
        print(f"Closed-loop MSE: {mse_closed_unscaled:.6f}")
        
        # Calculate oscillation-specific metrics
        print("\nCalculating oscillation-specific metrics...")
        osc_metrics_open, osc_summary_open = self.calculate_oscillation_metrics(y_test, y_pred_open)
        osc_metrics_closed, osc_summary_closed = self.calculate_oscillation_metrics(y_test, y_pred_closed)
        
        print("\nOscillation Metrics (Open-loop):")
        for key, value in osc_summary_open.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        print("\nOscillation Metrics (Closed-loop):")
        for key, value in osc_summary_closed.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # Prepare results dictionary
        results = {
            'true': all_true,
            'open_loop': all_open_loop,
            'closed_loop': all_closed_loop,
            'metrics': {
                'mse_open': float(mse_open),
                'mse_closed': float(mse_closed),
                'mae_open': float(mae_open),
                'mae_closed': float(mae_closed),
                'mse_open_unscaled': float(mse_open_unscaled),
                'mse_closed_unscaled': float(mse_closed_unscaled),
                'improvement': float(100 * (mse_open - mse_closed) / mse_open if mse_open > 0 else 0),
                'oscillation_metrics': {
                    'open_loop': osc_summary_open,
                    'closed_loop': osc_summary_closed
                }
            },
            'test_groups': [(float(g[0]), float(g[1])) for g in groups.keys()],
            'config': {
                'window_size': self.config['window_size'],
                'prediction_horizon': self.config['prediction_horizon'],
                'model_path': os.path.join(self.config['output_dir'], self.config['model_name'])
            },
            'scaler': scaler  # Save scaler for later use
        }
        
        return results
    
    def save_predictions(self, results, filename="predictions.pkl"):
        """Save prediction results to pickle file"""
        filepath = os.path.join(self.eval_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nPredictions saved to: {filepath}")
        return filepath
    
    def create_evaluation_plots(self, results):
        """Create comprehensive evaluation plots"""
        true_values = results['true']
        open_loop = results['open_loop']
        closed_loop = results['closed_loop']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Time series comparison
        ax1 = axes[0, 0]
        max_points = min(5000, len(true_values))
        ax1.plot(true_values[:max_points], 'b-', label='True β-value', linewidth=1.5)
        ax1.plot(open_loop[:max_points], 'r--', label='Open-Loop', linewidth=1.2, alpha=0.8)
        ax1.plot(closed_loop[:max_points], 'g-.', label='Closed-Loop', linewidth=1.2, alpha=0.8)
        ax1.set_title('Beta Value Predictions Comparison', fontsize=14)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('β-Value (scaled)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        error_open = open_loop - true_values
        error_closed = closed_loop - true_values
        
        ax2.hist(error_open, bins=50, alpha=0.5, label='Open-Loop Error', density=True)
        ax2.hist(error_closed, bins=50, alpha=0.5, label='Closed-Loop Error', density=True)
        ax2.set_title('Prediction Error Distribution', fontsize=14)
        ax2.set_xlabel('Error', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Oscillation range accuracy
        ax3 = axes[1, 0]
        if 'oscillation_metrics' in results['metrics']:
            osc_open = results['metrics']['oscillation_metrics']['open_loop']
            osc_closed = results['metrics']['oscillation_metrics']['closed_loop']
            
            categories = ['low_osc', 'medium_osc', 'high_osc']
            mse_open_vals = [osc_open.get(f'mse_{cat}', 0) for cat in categories]
            mse_closed_vals = [osc_closed.get(f'mse_{cat}', 0) for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax3.bar(x - width/2, mse_open_vals, width, label='Open-Loop', alpha=0.8)
            ax3.bar(x + width/2, mse_closed_vals, width, label='Closed-Loop', alpha=0.8)
            ax3.set_xlabel('Oscillation Range', fontsize=12)
            ax3.set_ylabel('MSE', fontsize=12)
            ax3.set_title('MSE by Oscillation Range', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(['Low', 'Medium', 'High'])
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Metrics summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics = results['metrics']
        metrics_text = f"Standard Metrics:\n"
        metrics_text += f"  Open-Loop MSE: {metrics['mse_open']:.6f}\n"
        metrics_text += f"  Closed-Loop MSE: {metrics['mse_closed']:.6f}\n"
        metrics_text += f"  Improvement: {metrics['improvement']:.1f}%\n\n"
        
        if 'oscillation_metrics' in metrics:
            osc_metrics = metrics['oscillation_metrics']['closed_loop']
            metrics_text += f"Oscillation Metrics (Closed-Loop):\n"
            if 'pp_ratio_mean' in osc_metrics:
                metrics_text += f"  Peak-to-Peak Ratio: {osc_metrics['pp_ratio_mean']:.3f} ± {osc_metrics['pp_ratio_std']:.3f}\n"
                metrics_text += f"  Within 20% of true: {osc_metrics['pp_ratio_within_20pct']:.1f}%\n"
            if 'freq_error_mean' in osc_metrics:
                metrics_text += f"  Frequency Error: {osc_metrics['freq_error_mean']:.3f} ± {osc_metrics['freq_error_std']:.3f}\n"
        
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(self.eval_dir, "evaluation_comprehensive.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional oscillation-focused plot
        self.create_oscillation_analysis_plot(results)
        
        print(f"Evaluation plots saved to: {self.eval_dir}")
    
    def create_oscillation_analysis_plot(self, results):
        """Create detailed oscillation analysis plot"""
        # Select sample sequences with different oscillation characteristics
        y_test = self.data['y_test']
        y_pred_open = self.model.predict(self.data['X_test'], verbose=0)
        
        # Find indices for low, medium, and high oscillation samples
        oscillation_ranges = []
        for i in range(len(y_test)):
            osc_range = y_test[i].max() - y_test[i].min()
            oscillation_ranges.append((i, osc_range))
        
        oscillation_ranges.sort(key=lambda x: x[1])
        
        # Select representative samples
        n_samples = len(oscillation_ranges)
        low_idx = oscillation_ranges[n_samples // 6][0]
        med_idx = oscillation_ranges[n_samples // 2][0]
        high_idx = oscillation_ranges[5 * n_samples // 6][0]
        
        # Create plot
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        sample_indices = [low_idx, med_idx, high_idx]
        titles = ['Low Oscillation', 'Medium Oscillation', 'High Oscillation']
        
        for i, (idx, title) in enumerate(zip(sample_indices, titles)):
            # Time domain plot
            ax_time = axes[i, 0]
            t = np.arange(len(y_test[idx]))
            ax_time.plot(t, y_test[idx], 'b-', label='True', linewidth=2)
            ax_time.plot(t, y_pred_open[idx], 'r--', label='Predicted', linewidth=2)
            ax_time.set_title(f'{title} - Time Domain', fontsize=12)
            ax_time.set_xlabel('Time Step')
            ax_time.set_ylabel('β-Value')
            ax_time.legend()
            ax_time.grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mse = np.mean((y_test[idx] - y_pred_open[idx])**2)
            pp_true = y_test[idx].max() - y_test[idx].min()
            pp_pred = y_pred_open[idx].max() - y_pred_open[idx].min()
            pp_ratio = pp_pred / pp_true if pp_true > 0 else 0
            
            ax_time.text(0.02, 0.98, f'MSE: {mse:.4f}\nPP Ratio: {pp_ratio:.2f}',
                        transform=ax_time.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Frequency domain plot (simple FFT)
            ax_freq = axes[i, 1]
            
            # Compute FFT
            fft_true = np.abs(np.fft.rfft(y_test[idx]))
            fft_pred = np.abs(np.fft.rfft(y_pred_open[idx]))
            freqs = np.fft.rfftfreq(len(y_test[idx]))
            
            ax_freq.plot(freqs[:20], fft_true[:20], 'b-', label='True', linewidth=2)
            ax_freq.plot(freqs[:20], fft_pred[:20], 'r--', label='Predicted', linewidth=2)
            ax_freq.set_title(f'{title} - Frequency Domain', fontsize=12)
            ax_freq.set_xlabel('Frequency (normalized)')
            ax_freq.set_ylabel('Magnitude')
            ax_freq.legend()
            ax_freq.grid(True, alpha=0.3)
        
        plt.suptitle('Oscillation Analysis by Range', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.eval_dir, "oscillation_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_parameter_grid_plots(self, results):
        """Create grid plots showing predictions for different amplitude-frequency combinations"""
        print("\nCreating parameter grid plots...")
        
        # Get test data and scaler
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        groups_test = self.data['groups_test']
        scaler = self.data['scaler']
        
        # Get predictions
        y_pred_open = self.model.predict(X_test, verbose=0)
        
        # Organize data by parameter groups
        unique_params = {}
        for i in range(len(X_test)):
            amp_scaled, freq_scaled = groups_test[i]
            key = (float(amp_scaled), float(freq_scaled))
            
            if key not in unique_params:
                unique_params[key] = {
                    'indices': [],
                    'sequences': []
                }
            unique_params[key]['indices'].append(i)
        
        # Sort keys for consistent ordering
        sorted_keys = sorted(unique_params.keys())
        
        # Determine grid size
        n_amps = 11  # 0.0 to 1.0 in steps of 0.1
        n_freqs = 10  # Different frequency values
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_amps, n_freqs, figsize=(n_freqs*3, n_amps*2.5))
        
        # Plot each parameter combination
        for idx, (amp_s, freq_s) in enumerate(sorted_keys):
            # Calculate grid position
            amp_idx = int(round(amp_s * (n_amps - 1)))
            freq_idx = int(round(freq_s * (n_freqs - 1)))
            
            ax = axes[amp_idx, freq_idx]
            
            # Get indices for this group
            indices = unique_params[(amp_s, freq_s)]['indices']
            
            if indices:
                # Collect predictions for this group
                group_true_scaled = []
                group_pred_scaled = []
                
                for i in indices:
                    # Collect scaled values
                    group_true_scaled.extend(y_test[i])
                    group_pred_scaled.extend(y_pred_open[i])
                
                # Convert to arrays
                group_true_scaled = np.array(group_true_scaled)
                group_pred_scaled = np.array(group_pred_scaled)
                
                # UNSCALE the beta values for plotting
                n_points = len(group_true_scaled)
                
                # Create dummy array with correct shape for inverse transform
                # scaler expects [amplitude, frequency, beta_value]
                dummy_data_true = np.column_stack([
                    np.full(n_points, amp_s),      # scaled amplitude
                    np.full(n_points, freq_s),      # scaled frequency  
                    group_true_scaled               # scaled beta values
                ])
                
                dummy_data_pred = np.column_stack([
                    np.full(n_points, amp_s),
                    np.full(n_points, freq_s),
                    group_pred_scaled
                ])
                
                # Inverse transform to get unscaled values
                unscaled_true_full = scaler.inverse_transform(dummy_data_true)
                unscaled_pred_full = scaler.inverse_transform(dummy_data_pred)
                
                # Extract unscaled values
                amp_unscaled = unscaled_true_full[0, 0]    # Actual amplitude in mA
                freq_unscaled = unscaled_true_full[0, 1]   # Actual frequency in Hz
                beta_true_unscaled = unscaled_true_full[:, 2]  # Actual beta values
                beta_pred_unscaled = unscaled_pred_full[:, 2]
                
                # Plot time series (limit points for clarity)
                max_points = min(200, len(beta_true_unscaled))
                time_steps = np.arange(max_points)
                
                ax.plot(time_steps, beta_true_unscaled[:max_points], 'b-', 
                    linewidth=1, alpha=0.8, label='True')
                ax.plot(time_steps, beta_pred_unscaled[:max_points], 'r--', 
                    linewidth=1, alpha=0.7, label='Pred')
                
                # Calculate MSE on UNSCALED data
                mse_unscaled = np.mean((beta_true_unscaled - beta_pred_unscaled)**2)
                
                # Also calculate MSE on scaled data for comparison
                mse_scaled = np.mean((group_true_scaled - group_pred_scaled)**2)
                
                # Format title with actual unscaled parameters
                ax.set_title(f'{amp_unscaled:.1f}mA, {freq_unscaled:.0f}Hz\n'
                            f'MSE={mse_unscaled:.4f}', 
                            fontsize=8, pad=2)
                
                # Set y-axis label for leftmost plots
                if freq_idx == 0:
                    ax.set_ylabel('Beta (unscaled)', fontsize=6)
                
                # Remove x-axis labels except for bottom row
                if amp_idx < n_amps - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('Time', fontsize=6)
                
                # Adjust y-axis ticks for readability
                ax.tick_params(axis='y', labelsize=6)
                ax.tick_params(axis='x', labelsize=6)
                
                # Add legend only to first subplot
                if amp_idx == 0 and freq_idx == 0:
                    ax.legend(fontsize=6, loc='upper right')
                    
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)
        
        # Add overall labels
        fig.text(0.5, 0.02, 'Frequency (Hz)', ha='center', fontsize=12)
        fig.text(0.02, 0.5, 'Amplitude (mA)', va='center', rotation='vertical', fontsize=12)
        
        plt.suptitle('Beta Predictions Grid by DBS Parameters (Unscaled Values)', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
        
        # Save plot
        grid_plot_path = os.path.join(self.eval_dir, "parameter_grid_unscaled.png")
        plt.savefig(grid_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter grid plot saved to: {grid_plot_path}")
        
        # Create MSE heatmap with unscaled MSE values
        self.create_mse_heatmap_unscaled(unique_params, scaler, X_test, y_test, y_pred_open)


    def create_mse_heatmap_unscaled(self, unique_params, scaler, X_test, y_test, y_pred_open):
        """Create a heatmap showing MSE for each parameter combination (using unscaled values)"""
        print("Creating MSE heatmap with unscaled values...")
        
        # Create MSE matrix
        n_amps = 11
        n_freqs = 10
        mse_matrix_unscaled = np.full((n_amps, n_freqs), np.nan)
        mse_matrix_scaled = np.full((n_amps, n_freqs), np.nan)
        
        # Also store actual parameter values
        amp_values_actual = np.full((n_amps, n_freqs), np.nan)
        freq_values_actual = np.full((n_amps, n_freqs), np.nan)
        
        # Calculate MSE for each group
        for (amp_s, freq_s), data in unique_params.items():
            indices = data['indices']
            if indices:
                # Collect all data for this group
                all_true_scaled = []
                all_pred_scaled = []
                
                for idx in indices:
                    all_true_scaled.append(y_test[idx])
                    all_pred_scaled.append(y_pred_open[idx])
                
                all_true_scaled = np.concatenate(all_true_scaled)
                all_pred_scaled = np.concatenate(all_pred_scaled)
                
                # Unscale the data
                n_points = len(all_true_scaled)
                dummy_data = np.column_stack([
                    np.full(n_points, amp_s),
                    np.full(n_points, freq_s),
                    all_true_scaled
                ])
                
                unscaled_data = scaler.inverse_transform(dummy_data)
                amp_actual = unscaled_data[0, 0]
                freq_actual = unscaled_data[0, 1]
                beta_true_unscaled = unscaled_data[:, 2]
                
                # Unscale predictions
                dummy_data[:, 2] = all_pred_scaled
                unscaled_pred = scaler.inverse_transform(dummy_data)
                beta_pred_unscaled = unscaled_pred[:, 2]
                
                # Calculate MSE on both scaled and unscaled data
                mse_unscaled = np.mean((beta_true_unscaled - beta_pred_unscaled)**2)
                mse_scaled = np.mean((all_true_scaled - all_pred_scaled)**2)
                
                # Place in matrix
                amp_idx = int(round(amp_s * (n_amps - 1)))
                freq_idx = int(round(freq_s * (n_freqs - 1)))
                
                mse_matrix_unscaled[amp_idx, freq_idx] = mse_unscaled
                mse_matrix_scaled[amp_idx, freq_idx] = mse_scaled
                amp_values_actual[amp_idx, freq_idx] = amp_actual
                freq_values_actual[amp_idx, freq_idx] = freq_actual
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Unscaled MSE
        im1 = ax1.imshow(mse_matrix_unscaled, cmap='RdYlGn_r', aspect='auto', 
                        interpolation='nearest')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('MSE (Unscaled Beta)', fontsize=12)
        
        # Set labels using actual parameter values
        # Get unique amp and freq values from the actual values matrix
        
        # For amplitudes (rows)
        amp_labels = []
        amp_indices_with_data = []
        for i in range(n_amps):
            row_values = amp_values_actual[i, :]
            valid_values = row_values[~np.isnan(row_values)]
            if len(valid_values) > 0:
                amp_labels.append(f'{valid_values[0]:.1f}')
                amp_indices_with_data.append(i)
        
        # For frequencies (columns)
        freq_labels = []
        freq_indices_with_data = []
        for j in range(n_freqs):
            col_values = freq_values_actual[:, j]
            valid_values = col_values[~np.isnan(col_values)]
            if len(valid_values) > 0:
                freq_labels.append(f'{valid_values[0]:.0f}')
                freq_indices_with_data.append(j)
        
        # Set ticks only where we have data
        ax1.set_xticks(freq_indices_with_data)
        ax1.set_xticklabels(freq_labels, rotation=45)
        ax1.set_yticks(amp_indices_with_data)
        ax1.set_yticklabels(amp_labels)
        
        # Add value annotations
        for i in range(n_amps):
            for j in range(n_freqs):
                if not np.isnan(mse_matrix_unscaled[i, j]):
                    text = ax1.text(j, i, f'{mse_matrix_unscaled[i, j]:.4f}',
                                ha="center", va="center", 
                                color="white" if mse_matrix_unscaled[i, j] > np.nanmean(mse_matrix_unscaled) else "black",
                                fontsize=7)
        
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Amplitude (mA)', fontsize=12)
        ax1.set_title('MSE Heatmap (Unscaled Beta Values)', fontsize=14)
        
        # Plot 2: Scaled MSE for comparison
        im2 = ax2.imshow(mse_matrix_scaled, cmap='RdYlGn_r', aspect='auto', 
                        interpolation='nearest')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('MSE (Scaled Beta)', fontsize=12)
        
        # Use same tick positions and labels
        ax2.set_xticks(freq_indices_with_data)
        ax2.set_xticklabels(freq_labels, rotation=45)
        ax2.set_yticks(amp_indices_with_data)
        ax2.set_yticklabels(amp_labels)
        
        # Add value annotations
        for i in range(n_amps):
            for j in range(n_freqs):
                if not np.isnan(mse_matrix_scaled[i, j]):
                    text = ax2.text(j, i, f'{mse_matrix_scaled[i, j]:.4f}',
                                ha="center", va="center", 
                                color="white" if mse_matrix_scaled[i, j] > np.nanmean(mse_matrix_scaled) else "black",
                                fontsize=7)
        
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Amplitude (mA)', fontsize=12)
        ax2.set_title('MSE Heatmap (Scaled Beta Values)', fontsize=14)
        
        plt.tight_layout()
        heatmap_path = os.path.join(self.eval_dir, "mse_heatmap_comparison.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MSE heatmap saved to: {heatmap_path}")
        
        # Print summary statistics
        print(f"\nMSE Summary:")
        print(f"  Unscaled - Min: {np.nanmin(mse_matrix_unscaled):.6f}, "
            f"Max: {np.nanmax(mse_matrix_unscaled):.6f}, "
            f"Mean: {np.nanmean(mse_matrix_unscaled):.6f}")
        print(f"  Scaled - Min: {np.nanmin(mse_matrix_scaled):.6f}, "
            f"Max: {np.nanmax(mse_matrix_scaled):.6f}, "
            f"Mean: {np.nanmean(mse_matrix_scaled):.6f}")
        
        # Print which parameter combinations were found
        print(f"\nFound {len(amp_labels)} unique amplitude values: {amp_labels}")
        print(f"Found {len(freq_labels)} unique frequency values: {freq_labels}")

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting model evaluation...")
        
        # Generate predictions
        results = self.generate_closed_loop_predictions()
        
        # Save predictions
        pkl_path = self.save_predictions(results, "all_predictions.pkl")
        
        # Create plots
        self.create_evaluation_plots(results)
        
        # Create parameter grid plots - ADD THIS LINE
        self.create_parameter_grid_plots(results)
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.eval_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save oscillation analysis report
        self.save_oscillation_report(results)
        
        print(f"\nEvaluation complete!")
        print(f"Predictions saved to: {pkl_path}")
        print(f"Metrics saved to: {metrics_path}")
        
        return results['metrics']
    
    def save_oscillation_report(self, results):
        """Save detailed oscillation analysis report"""
        report_path = os.path.join(self.eval_dir, "oscillation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BETA OSCILLATION MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.config['model_name']}\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            metrics = results['metrics']
            
            f.write("STANDARD METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Open-Loop MSE:    {metrics['mse_open']:.6f}\n")
            f.write(f"Closed-Loop MSE:  {metrics['mse_closed']:.6f}\n")
            f.write(f"Open-Loop MAE:    {metrics['mae_open']:.6f}\n")
            f.write(f"Closed-Loop MAE:  {metrics['mae_closed']:.6f}\n")
            f.write(f"Improvement:      {metrics['improvement']:.1f}%\n\n")
            
            if 'oscillation_metrics' in metrics:
                f.write("OSCILLATION-SPECIFIC METRICS\n")
                f.write("-" * 30 + "\n")
                
                for loop_type in ['open_loop', 'closed_loop']:
                    f.write(f"\n{loop_type.upper().replace('_', '-')}:\n")
                    osc_metrics = metrics['oscillation_metrics'][loop_type]
                    
                    if 'pp_ratio_mean' in osc_metrics:
                        f.write(f"  Peak-to-Peak Ratio:     {osc_metrics['pp_ratio_mean']:.3f} ± {osc_metrics['pp_ratio_std']:.3f}\n")
                        f.write(f"  PP Ratio Median:        {osc_metrics['pp_ratio_median']:.3f}\n")
                        f.write(f"  Within 20% of true:     {osc_metrics['pp_ratio_within_20pct']:.1f}%\n")
                    
                    if 'freq_error_mean' in osc_metrics:
                        f.write(f"  Frequency Error:        {osc_metrics['freq_error_mean']:.3f} ± {osc_metrics['freq_error_std']:.3f}\n")
                    
                    f.write("\n  MSE by Oscillation Range:\n")
                    for range_type in ['low', 'medium', 'high']:
                        key = f'mse_{range_type}_osc'
                        if key in osc_metrics:
                            n_samples = osc_metrics.get(f'n_samples_{range_type}_osc', 0)
                            f.write(f"    {range_type.capitalize():8s}: {osc_metrics[key]:.6f} ({n_samples} samples)\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("END OF REPORT\n")
        
        print(f"Oscillation report saved to: {report_path}")