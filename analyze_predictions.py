#!/usr/bin/env python3
"""
Analyze saved predictions from the beta oscillation model
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_predictions(pkl_path):
    """Load predictions from pickle file"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results

def analyze_predictions(results):
    """Perform detailed analysis of predictions"""
    print("=== Prediction Analysis ===\n")
    
    # Get data
    true_values = results['true']
    open_loop = results['open_loop']
    closed_loop = results['closed_loop']
    
    # Basic statistics
    print(f"Data shape: {len(true_values)} predictions")
    print(f"Value ranges (scaled):")
    print(f"  True: [{np.min(true_values):.4f}, {np.max(true_values):.4f}]")
    print(f"  Open-loop: [{np.min(open_loop):.4f}, {np.max(open_loop):.4f}]")
    print(f"  Closed-loop: [{np.min(closed_loop):.4f}, {np.max(closed_loop):.4f}]")
    
    # If scaler is available, show unscaled ranges
    if 'scaler' in results:
        scaler = results['scaler']
        # Create dummy arrays for inverse transform
        n_points = 1000  # Sample for efficiency
        sample_indices = np.random.choice(len(true_values), min(n_points, len(true_values)), replace=False)
        
        dummy_true = np.column_stack([
            np.zeros(len(sample_indices)),
            np.zeros(len(sample_indices)),
            true_values[sample_indices]
        ])
        dummy_open = dummy_true.copy()
        dummy_open[:, 2] = open_loop[sample_indices]
        dummy_closed = dummy_true.copy()
        dummy_closed[:, 2] = closed_loop[sample_indices]
        
        unscaled_true = scaler.inverse_transform(dummy_true)[:, 2]
        unscaled_open = scaler.inverse_transform(dummy_open)[:, 2]
        unscaled_closed = scaler.inverse_transform(dummy_closed)[:, 2]
        
        print(f"\nValue ranges (unscaled):")
        print(f"  True: [{np.min(unscaled_true):.4f}, {np.max(unscaled_true):.4f}]")
        print(f"  Open-loop: [{np.min(unscaled_open):.4f}, {np.max(unscaled_open):.4f}]")
        print(f"  Closed-loop: [{np.min(unscaled_closed):.4f}, {np.max(unscaled_closed):.4f}]")
        
        # Calculate proper unscaled MSE
        mse_open_unscaled = np.mean((unscaled_open - unscaled_true)**2)
        mse_closed_unscaled = np.mean((unscaled_closed - unscaled_true)**2)
        print(f"\nProper MSE (unscaled):")
        print(f"  Open-loop: {mse_open_unscaled:.6f}")
        print(f"  Closed-loop: {mse_closed_unscaled:.6f}")
    
    # Error analysis
    print("\n=== Error Analysis ===")
    errors_open = open_loop - true_values
    errors_closed = closed_loop - true_values
    
    print(f"Mean Absolute Error:")
    print(f"  Open-loop: {np.mean(np.abs(errors_open)):.6f}")
    print(f"  Closed-loop: {np.mean(np.abs(errors_closed)):.6f}")
    
    print(f"Error std:")
    print(f"  Open-loop: {np.std(errors_open):.6f}")
    print(f"  Closed-loop: {np.std(errors_closed):.6f}")
    
    # Percentile analysis
    print(f"\nError percentiles:")
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        print(f"  {p}th percentile - Open: {np.percentile(np.abs(errors_open), p):.6f}, "
              f"Closed: {np.percentile(np.abs(errors_closed), p):.6f}")
    
    # Oscillation metrics from saved results
    if 'metrics' in results:
        print("\n=== Saved Metrics ===")
        metrics = results['metrics']
        print(f"MSE Open-loop: {metrics['mse_open']:.6f}")
        print(f"MSE Closed-loop: {metrics['mse_closed']:.6f}")
        print(f"Improvement: {metrics['improvement']:.1f}%")
        
        if 'oscillation_metrics' in metrics:
            osc_metrics = metrics['oscillation_metrics']['closed_loop']
            print(f"\nClosed-loop oscillation metrics:")
            print(f"  Peak-to-peak ratio: {osc_metrics.get('pp_ratio_mean', 'N/A'):.3f}")
            print(f"  Within 20% of true: {osc_metrics.get('pp_ratio_within_20pct', 'N/A'):.1f}%")
    
    return true_values, open_loop, closed_loop

def create_analysis_plots(true_values, open_loop, closed_loop, output_dir):
    """Create detailed analysis plots"""
    
    # 1. Error distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error histograms
    ax = axes[0, 0]
    errors_open = open_loop - true_values
    errors_closed = closed_loop - true_values
    
    ax.hist(errors_open, bins=50, alpha=0.5, label='Open-loop', density=True)
    ax.hist(errors_closed, bins=50, alpha=0.5, label='Closed-loop', density=True)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[0, 1]
    ax.scatter(true_values[:1000], open_loop[:1000], alpha=0.5, s=1, label='Open-loop')
    ax.scatter(true_values[:1000], closed_loop[:1000], alpha=0.5, s=1, label='Closed-loop')
    ax.plot([true_values.min(), true_values.max()], 
            [true_values.min(), true_values.max()], 'k--', alpha=0.5)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction vs True (first 1000 points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time series comparison
    ax = axes[1, 0]
    n_points = min(500, len(true_values))
    ax.plot(true_values[:n_points], 'k-', label='True', linewidth=1.5)
    ax.plot(open_loop[:n_points], 'b--', label='Open-loop', alpha=0.7)
    ax.plot(closed_loop[:n_points], 'r:', label='Closed-loop', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Beta Value')
    ax.set_title('Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative error
    ax = axes[1, 1]
    cum_error_open = np.cumsum(np.abs(errors_open))
    cum_error_closed = np.cumsum(np.abs(errors_closed))
    x = np.arange(len(cum_error_open))
    ax.plot(x, cum_error_open, label='Open-loop')
    ax.plot(x, cum_error_closed, label='Closed-loop')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Absolute Error')
    ax.set_title('Error Accumulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300)
    plt.close()
    
    # 2. Oscillation analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calculate rolling statistics
    window = 50
    
    # Rolling range (peak-to-peak)
    ax = axes[0, 0]
    true_pp = np.array([true_values[i:i+window].max() - true_values[i:i+window].min() 
                       for i in range(0, len(true_values)-window, 10)])
    open_pp = np.array([open_loop[i:i+window].max() - open_loop[i:i+window].min() 
                       for i in range(0, len(open_loop)-window, 10)])
    closed_pp = np.array([closed_loop[i:i+window].max() - closed_loop[i:i+window].min() 
                         for i in range(0, len(closed_loop)-window, 10)])
    
    x_roll = np.arange(0, len(true_values)-window, 10)
    ax.plot(x_roll, true_pp, 'k-', label='True')
    ax.plot(x_roll, open_pp, 'b--', label='Open-loop', alpha=0.7)
    ax.plot(x_roll, closed_pp, 'r:', label='Closed-loop', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Peak-to-Peak Range')
    ax.set_title(f'Rolling Oscillation Range (window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Peak-to-peak ratio scatter
    ax = axes[0, 1]
    pp_ratio_open = open_pp / (true_pp + 1e-6)
    pp_ratio_closed = closed_pp / (true_pp + 1e-6)
    
    ax.scatter(true_pp, pp_ratio_open, alpha=0.5, s=10, label='Open-loop')
    ax.scatter(true_pp, pp_ratio_closed, alpha=0.5, s=10, label='Closed-loop')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0.8, color='r', linestyle=':', alpha=0.5)
    ax.axhline(y=1.2, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('True Peak-to-Peak Range')
    ax.set_ylabel('Predicted/True Ratio')
    ax.set_title('Oscillation Amplitude Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3])
    
    # Error vs oscillation magnitude
    ax = axes[1, 0]
    # Group by oscillation magnitude
    true_ranges = []
    open_errors = []
    closed_errors = []
    
    for i in range(0, len(true_values)-window, window):
        tr = true_values[i:i+window].max() - true_values[i:i+window].min()
        oe = np.mean(np.abs(errors_open[i:i+window]))
        ce = np.mean(np.abs(errors_closed[i:i+window]))
        true_ranges.append(tr)
        open_errors.append(oe)
        closed_errors.append(ce)
    
    ax.scatter(true_ranges, open_errors, alpha=0.5, s=20, label='Open-loop')
    ax.scatter(true_ranges, closed_errors, alpha=0.5, s=20, label='Closed-loop')
    ax.set_xlabel('True Oscillation Range')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Error vs Oscillation Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Performance Summary\n" + "="*30 + "\n\n"
    
    # Calculate metrics
    open_within_20 = np.sum((pp_ratio_open > 0.8) & (pp_ratio_open < 1.2)) / len(pp_ratio_open) * 100
    closed_within_20 = np.sum((pp_ratio_closed > 0.8) & (pp_ratio_closed < 1.2)) / len(pp_ratio_closed) * 100
    
    summary_text += f"Peak-to-Peak Accuracy:\n"
    summary_text += f"  Open-loop: {open_within_20:.1f}% within ±20%\n"
    summary_text += f"  Closed-loop: {closed_within_20:.1f}% within ±20%\n\n"
    
    summary_text += f"Mean Absolute Error:\n"
    summary_text += f"  Open-loop: {np.mean(np.abs(errors_open)):.6f}\n"
    summary_text += f"  Closed-loop: {np.mean(np.abs(errors_closed)):.6f}\n\n"
    
    summary_text += f"Error Std Dev:\n"
    summary_text += f"  Open-loop: {np.std(errors_open):.6f}\n"
    summary_text += f"  Closed-loop: {np.std(errors_closed):.6f}\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='center', 
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oscillation_analysis.png'), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze beta oscillation model predictions')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to predictions pickle file')
    parser.add_argument('--output_dir', type=str, default='analysis_plots',
                        help='Directory to save analysis plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions_file}")
    results = load_predictions(args.predictions_file)
    
    # Analyze predictions
    true_values, open_loop, closed_loop = analyze_predictions(results)
    
    # Create plots
    print("\nCreating analysis plots...")
    create_analysis_plots(true_values, open_loop, closed_loop, args.output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()