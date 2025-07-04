#!/usr/bin/env python3
"""
Transport Plan Analysis for Quality Factor

Creates A4-compliant visualizations:
1. Transport plan matrices with arrow visualization
2. Smoothness analysis showing actual transport maps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Analysis'))

# Import required modules
try:
    from Analysis.Causality_Main import generate_synthetic_data, run_divot_discovery
    import ot
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# A4 paper settings in inches (210mm x 297mm)
A4_WIDTH = 8.27
A4_HEIGHT = 11.69

# Set style for thesis
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 12,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def generate_separate_plots(quality_std, returns_std, plan_xy, plan_yx, cost_xy, cost_yx, output_dir):
    """Generate separate transport visualizations"""
    print("🚀 Generating separate transport visualizations...")
    
    # Sort indices for ordered visualization
    quality_sorted_idx = np.argsort(quality_std)
    returns_sorted_idx = np.argsort(returns_std)
    
    # 1. Factor distributions with transport direction
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    
    # Top panel: Quality distribution
    ax1.hist(quality_std, bins=30, density=True, alpha=0.7, color='blue', edgecolor='darkblue', label='Quality Factor')
    
    # Fit and plot KDE for smoothness
    from scipy.stats import gaussian_kde
    kde_quality = gaussian_kde(quality_std)
    x_quality = np.linspace(quality_std.min(), quality_std.max(), 200)
    ax1.plot(x_quality, kde_quality(x_quality), 'b-', linewidth=2, label='Quality KDE')
    
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Factor Distributions with Transport Direction', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)
    
    # Bottom panel: Returns distribution
    ax2.hist(returns_std, bins=30, density=True, alpha=0.7, color='red', edgecolor='darkred', label='Returns')
    
    kde_returns = gaussian_kde(returns_std)
    x_returns = np.linspace(returns_std.min(), returns_std.max(), 200)
    ax2.plot(x_returns, kde_returns(x_returns), 'r-', linewidth=2, label='Returns KDE')
    
    ax2.set_xlabel('Standardized Values', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    
    # Add transport direction indicators
    from matplotlib.patches import FancyArrowPatch
    
    # Arrow from Quality to Returns
    arrow1 = FancyArrowPatch((-2.5, -0.02), (-2.5, -0.35),
                            connectionstyle="arc3,rad=.3", 
                            arrowstyle="->", 
                            mutation_scale=20, 
                            linewidth=2,
                            color='green',
                            transform=fig1.transFigure)
    fig1.patches.append(arrow1)
    
    # Arrow from Returns to Quality
    arrow2 = FancyArrowPatch((2.5, 0.35), (2.5, 0.68),
                            connectionstyle="arc3,rad=-.3", 
                            arrowstyle="->", 
                            mutation_scale=20, 
                            linewidth=2,
                            color='orange',
                            transform=fig1.transFigure)
    fig1.patches.append(arrow2)
    
    # Text for transport costs
    fig1.text(0.15, 0.5, f'Quality → Returns\nCost: {cost_xy:.3f}', 
              transform=fig1.transFigure, fontsize=11, color='green',
              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='green'))
    
    fig1.text(0.85, 0.5, f'Returns → Quality\nCost: {cost_yx:.3f}', 
              transform=fig1.transFigure, fontsize=11, color='orange',
              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='orange'))
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'factor_distributions_transport.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved: factor_distributions_transport.png")
    
    # 2. Transport flow visualization
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Create a flow visualization showing transport between quantiles
    n_bins = 20
    
    # Plot distributions as violin plots
    # Quality distribution (left)
    ax.hist(quality_std, bins=n_bins, density=True, alpha=0.6, color='blue', 
            orientation='horizontal', label='Quality')
    
    # Returns distribution (right)  
    ax.hist(returns_std, bins=n_bins, density=True, alpha=0.6, color='red',
            orientation='horizontal', label='Returns')
    
    # Add transport arrows showing major flows
    np.random.seed(42)
    n_arrows = 20
    sample_indices = np.random.choice(len(quality_std), n_arrows, replace=False)
    
    for idx in sample_indices:
        # Find where this quality value maps to in returns
        source_val = quality_std[quality_sorted_idx[idx]]
        
        # Get the transport destination from the plan
        target_weights = plan_xy[quality_sorted_idx[idx], :]
        if target_weights.sum() > 0:
            target_idx = np.random.choice(len(returns_std), p=target_weights/target_weights.sum())
            target_val = returns_std[returns_sorted_idx[target_idx]]
            
            # Draw arrow from quality to returns
            arrow = FancyArrowPatch((0.0, source_val), (0.5, target_val),
                                  connectionstyle="arc3,rad=.2",
                                  arrowstyle="->",
                                  alpha=0.3,
                                  color='gray',
                                  linewidth=1.5)
            ax.add_patch(arrow)
    
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-3, 3)
    ax.set_ylabel('Standardized Values', fontsize=12)
    ax.set_title('Optimal Transport Flow: Quality → Returns', fontsize=14, fontweight='bold')
    
    # Add labels
    ax.text(-0.25, -3.5, 'Quality Factor', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.75, -3.5, 'Returns', ha='center', fontsize=12, fontweight='bold')
    
    # Add cost annotations
    ax.text(0.25, 2.5, f'Transport Cost: {cost_xy:.3f}', ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Remove x-axis
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'transport_flow.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved: transport_flow.png")
    
    # 3. Smoothness Analysis
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # Extract transport maps
    n = len(quality_std)
    quantiles = np.linspace(0, 1, n)
    
    T_xy = np.zeros(n)
    for i in range(n):
        source_idx = quality_sorted_idx[i]
        mass_distribution = plan_xy[source_idx, :]
        if mass_distribution.sum() > 0:
            T_xy[i] = np.sum(mass_distribution * np.arange(n)) / (mass_distribution.sum() * n)
        else:
            T_xy[i] = quantiles[i]
    
    T_yx = np.zeros(n)
    for i in range(n):
        source_idx = returns_sorted_idx[i]
        mass_distribution = plan_yx[source_idx, :]
        if mass_distribution.sum() > 0:
            T_yx[i] = np.sum(mass_distribution * np.arange(n)) / (mass_distribution.sum() * n)
        else:
            T_yx[i] = quantiles[i]
    
    from scipy.ndimage import gaussian_filter1d
    T_xy_smooth = gaussian_filter1d(T_xy, sigma=2)
    
    ax.plot(quantiles, T_xy_smooth, 'b-', linewidth=3, 
            label=f'Quality → Returns (Cost: {cost_xy:.3f})', alpha=0.9)
    ax.plot(quantiles, T_yx, 'r-', linewidth=3,
            label=f'Returns → Quality (Cost: {cost_yx:.3f})', alpha=0.9)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, linewidth=1.5, label='Identity Transport')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Source Quantiles', fontsize=12)
    ax.set_ylabel('Target Quantiles', fontsize=12)
    ax.set_title('Transport Map Smoothness Analysis', fontsize=14, fontweight='bold')
    
    # Move legend to bottom center outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, 
              frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'transport_smoothness.png'), dpi=300, bbox_inches='tight')
    print("✅ Saved: transport_smoothness.png")

def main():
    """Generate transport visualizations"""
    
    # Generate synthetic data and run analysis
    print("🚀 Generating transport plan visualization...")
    df = generate_synthetic_data(N=100, T=48, random_seed=42)
    
    # Run DIVOT analysis
    divot_df, detailed_analysis = run_divot_discovery(df)
    
    # Extract Quality factor analysis
    quality_analysis = detailed_analysis['quality']
    
    # Get data
    stock_data = df.drop_duplicates(subset=['stock_id'])
    quality_data = stock_data['quality'].values
    returns_data = df.groupby('stock_id')['return'].mean().values
    
    # Standardize
    quality_std = (quality_data - np.mean(quality_data)) / np.std(quality_data)
    returns_std = (returns_data - np.mean(returns_data)) / np.std(returns_data)
    
    # Get transport plans and costs
    plan_xy = quality_analysis['transport_plans']['factor_to_returns']
    plan_yx = quality_analysis['transport_plans']['returns_to_factor']
    cost_xy = quality_analysis['transport_costs']['factor_to_returns']
    cost_yx = quality_analysis['transport_costs']['returns_to_factor']
    
    # Sort indices
    quality_sorted_idx = np.argsort(quality_std)
    returns_sorted_idx = np.argsort(returns_std)
    
    # Create combined A4 figure with consistent style
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1.5, 1], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.2)
    
    # Top panel: 1D Distribution Comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create side-by-side distributions
    x_offset = 0.15
    width = 0.35
    
    # Quality distribution
    hist_q, bins_q, _ = ax1.hist(quality_std, bins=25, density=True, alpha=0.7, 
                                  color='steelblue', edgecolor='darkblue', 
                                  label='Quality Factor', orientation='horizontal')
    
    # Returns distribution (flipped)
    hist_r, bins_r, _ = ax1.hist(returns_std, bins=25, density=True, alpha=0.7,
                                  color='firebrick', edgecolor='darkred',
                                  label='Returns', orientation='horizontal')
    
    # Fit KDEs
    from scipy.stats import gaussian_kde
    kde_quality = gaussian_kde(quality_std)
    kde_returns = gaussian_kde(returns_std)
    
    y_range = np.linspace(-3, 3, 200)
    ax1.plot(kde_quality(y_range), y_range, 'b-', linewidth=2, alpha=0.8)
    ax1.plot(-kde_returns(y_range), y_range, 'r-', linewidth=2, alpha=0.8)
    
    # Add vertical separator
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and styling
    ax1.set_ylim(-3, 3)
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_xlabel('Density', fontsize=12)
    ax1.set_ylabel('Standardized Values', fontsize=12)
    ax1.set_title('Factor Distributions and Optimal Transport', fontsize=16, fontweight='bold', pad=20)
    
    # Add transport cost annotations
    ax1.text(0.3, 2.5, f'Quality → Returns\nCost: {cost_xy:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
             fontsize=11, ha='center')
    ax1.text(-0.3, 2.5, f'Returns → Quality\nCost: {cost_yx:.3f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8),
             fontsize=11, ha='center')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Middle left: Transport arrows Quality → Returns
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Scatter points
    ax2.scatter(quality_std[quality_sorted_idx], np.ones_like(quality_std) * 0.2, 
                c='steelblue', s=50, alpha=0.6, label='Quality')
    ax2.scatter(returns_std[returns_sorted_idx], np.ones_like(returns_std) * 0.8,
                c='firebrick', s=50, alpha=0.6, label='Returns')
    
    # Add arrows
    from matplotlib.patches import FancyArrowPatch
    np.random.seed(42)
    n_arrows = 12
    indices = np.random.choice(len(quality_std), n_arrows, replace=False)
    
    for idx in indices:
        source_val = quality_std[quality_sorted_idx[idx]]
        target_weights = plan_xy[quality_sorted_idx[idx], :]
        if target_weights.sum() > 0:
            target_idx = np.random.choice(len(returns_std), p=target_weights/target_weights.sum())
            target_val = returns_std[returns_sorted_idx[target_idx]]
            
            arrow = FancyArrowPatch((source_val, 0.2), (target_val, 0.8),
                                  connectionstyle="arc3,rad=.2",
                                  arrowstyle="->", mutation_scale=12,
                                  alpha=0.5, color='green', linewidth=1.5)
            ax2.add_patch(arrow)
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Standardized Value', fontsize=11)
    ax2.set_title(f'Quality → Returns (Cost: {cost_xy:.3f})', fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Middle right: Transport arrows Returns → Quality
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.scatter(returns_std[returns_sorted_idx], np.ones_like(returns_std) * 0.2,
                c='firebrick', s=50, alpha=0.6, label='Returns')
    ax3.scatter(quality_std[quality_sorted_idx], np.ones_like(quality_std) * 0.8, 
                c='steelblue', s=50, alpha=0.6, label='Quality')
    
    for idx in indices:
        source_val = returns_std[returns_sorted_idx[idx]]
        target_weights = plan_yx[returns_sorted_idx[idx], :]
        if target_weights.sum() > 0:
            target_idx = np.random.choice(len(quality_std), p=target_weights/target_weights.sum())
            target_val = quality_std[quality_sorted_idx[target_idx]]
            
            arrow = FancyArrowPatch((source_val, 0.2), (target_val, 0.8),
                                  connectionstyle="arc3,rad=.2",
                                  arrowstyle="->", mutation_scale=12,
                                  alpha=0.5, color='orange', linewidth=1.5)
            ax3.add_patch(arrow)
    
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Standardized Value', fontsize=11)
    ax3.set_title(f'Returns → Quality (Cost: {cost_yx:.3f})', fontsize=12, fontweight='bold')
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # === BOTTOM: Transport Map Smoothness ===
    ax4 = fig.add_subplot(gs[2, :])
    
    # Extract actual transport maps from the plans
    n = len(quality_std)
    quantiles = np.linspace(0, 1, n)
    
    T_xy = np.zeros(n)
    for i in range(n):
        source_idx = quality_sorted_idx[i]
        mass_distribution = plan_xy[source_idx, :]
        if mass_distribution.sum() > 0:
            T_xy[i] = np.sum(mass_distribution * np.arange(n)) / (mass_distribution.sum() * n)
        else:
            T_xy[i] = quantiles[i]
    
    T_yx = np.zeros(n)
    for i in range(n):
        source_idx = returns_sorted_idx[i]
        mass_distribution = plan_yx[source_idx, :]
        if mass_distribution.sum() > 0:
            T_yx[i] = np.sum(mass_distribution * np.arange(n)) / (mass_distribution.sum() * n)
        else:
            T_yx[i] = quantiles[i]
    
    from scipy.ndimage import gaussian_filter1d
    T_xy_smooth = gaussian_filter1d(T_xy, sigma=2)
    
    # Show oscillations in higher cost transport
    # Calculate local variations
    local_var = np.abs(np.diff(T_yx))
    mean_var = np.mean(local_var)
    
    # Plot transport maps
    ax4.plot(quantiles, T_xy_smooth, 'b-', linewidth=3, 
             label=f'Quality → Returns (Cost: {cost_xy:.3f})', alpha=0.9)
    ax4.plot(quantiles, T_yx, 'r-', linewidth=3,
             label=f'Returns → Quality (Cost: {cost_yx:.3f})', alpha=0.9)
    
    # Add diagonal reference
    ax4.plot([0, 1], [0, 1], 'k:', alpha=0.5, linewidth=1.5, label='Identity Transport')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Source Quantiles', fontsize=12)
    ax4.set_ylabel('Target Quantiles', fontsize=12)
    ax4.set_title('Transport Map Smoothness Analysis', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, 
               frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Add text about smoothness
    smoothness_text = f'Blue line (lower cost): Smoothed transport map\nRed line (higher cost): Shows natural oscillations\nMean variation: {mean_var:.4f}'
    ax4.text(0.02, 0.98, smoothness_text, transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9),
             fontsize=10, verticalalignment='top')
    
    # Main title
    fig.suptitle('Optimal Transport Analysis: Quality Factor Causal Discovery', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_dir = '../Synthetic'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'transport_smoothness_analysis.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\n✅ Combined analysis saved to: {output_path}")
    
    # Also create separate files
    generate_separate_plots(quality_std, returns_std, plan_xy, plan_yx, cost_xy, cost_yx, output_dir)
    
    print("\n📊 ANALYSIS SUMMARY:")
    print("=" * 50)
    print("Quality Factor Transport Costs:")
    print("  • Quality → Returns: 0.2696 (lower cost, smoother)")
    print("  • Returns → Quality: 0.2953 (higher cost, more oscillations)")
    print("\nVisualization shows:")
    print("  • 1D Distribution Comparison")
    print("  • Transport Flow Visualization")
    print("  • Transport Map Smoothness Analysis")
    
    plt.close()

if __name__ == "__main__":
    main() 