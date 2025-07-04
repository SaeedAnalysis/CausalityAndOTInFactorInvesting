#!/usr/bin/env python3
"""
Generate Synthetic Data Correlation Analysis Figure

Creates a comprehensive 4-panel figure showing:
- Top left: Correlation matrix between factors and returns
- Top right: True causal effects vs observed correlations comparison
- Bottom left: Quality-returns scatter plot with trend line
- Bottom right: Summary statistics table

Academic styling: no titles, serif fonts, high resolution.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Add Analysis directory to path
analysis_dir = Path(__file__).parent.parent.parent / "Analysis"
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

from Causality_Main import generate_synthetic_data

# Academic plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'mathtext.fontset': 'cm'
})

def load_metrics():
    """Load latest metrics with fallback."""
    metrics_file = Path(__file__).parent.parent.parent / "latest_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    else:
        # Fallback metrics based on known values
        return {
            'synthetic_seed_value': 42,
            'true_treatment_effect': 0.05,
            'true_quality_effect': 0.01,
            'true_size_effect': 0.005,
            'true_volatility_effect': -0.005,
            'true_value_effect': 0.0
        }

def create_correlation_matrix_panel(ax, df):
    """Create correlation matrix panel (top left)."""
    # Calculate correlation matrix - use actual column names
    factor_cols = ['value', 'size', 'quality', 'volatility', 'return']
    
    # Check if columns exist, if not, try alternative names
    available_cols = []
    for col in factor_cols:
        if col in df.columns:
            available_cols.append(col)
        elif f'{col}_factor' in df.columns:
            available_cols.append(f'{col}_factor')
    
    corr_matrix = df[available_cols].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                ax=ax)
    
    # Clean up labels - use available columns
    clean_labels = [col.title() for col in available_cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right')
    ax.set_yticklabels(clean_labels, rotation=0)
    
    # Add panel label
    ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

def create_effects_comparison_panel(ax, df, metrics):
    """Create true vs observed effects comparison (top right)."""
    # True causal effects
    true_effects = {
        'Quality': metrics.get('true_quality_effect', 0.01),
        'Size': metrics.get('true_size_effect', 0.005),
        'Volatility': metrics.get('true_volatility_effect', -0.005),
        'Value': metrics.get('true_value_effect', 0.0)
    }
    
    # Calculate observed correlations with returns
    factor_cols = ['value', 'size', 'quality', 'volatility']
    factor_names = ['Value', 'Size', 'Quality', 'Volatility']
    
    observed_corrs = {}
    for col, name in zip(factor_cols, factor_names):
        if col in df.columns:
            observed_corrs[name] = df[col].corr(df['return'])
        else:
            observed_corrs[name] = 0.0  # Fallback
    
    # Create comparison
    factors = list(true_effects.keys())
    true_vals = [true_effects[f] for f in factors]
    observed_vals = [observed_corrs[f] for f in factors]
    
    x = np.arange(len(factors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, true_vals, width, label='True Causal Effect', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, observed_vals, width, label='Observed Correlation',
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Factor')
    ax.set_ylabel('Effect Size')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend(frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.002),
                f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.002),
                f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Add panel label
    ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

def create_quality_scatter_panel(ax, df):
    """Create quality-returns scatter plot (bottom left)."""
    # Sample data for readability (use every 10th point)
    sample_df = df.iloc[::10].copy()
    
    # Use correct column name
    quality_col = 'quality' if 'quality' in df.columns else 'quality_factor'
    
    # Create scatter plot
    ax.scatter(sample_df[quality_col], sample_df['return'], 
               alpha=0.6, s=20, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(sample_df[quality_col], sample_df['return'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(sample_df[quality_col].min(), sample_df[quality_col].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # Calculate and display R²
    correlation = sample_df[quality_col].corr(sample_df['return'])
    r_squared = correlation**2
    
    ax.text(0.05, 0.95, f'R² = {r_squared:.3f}\nρ = {correlation:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Quality Factor')
    ax.set_ylabel('Return')
    ax.legend(frameon=True, fancybox=False, shadow=False, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add panel label
    ax.text(-0.15, 1.05, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

def create_summary_table_panel(ax, df, metrics):
    """Create summary statistics table (bottom right)."""
    ax.axis('off')
    
    # Calculate comprehensive statistics - use actual column names
    factor_cols = ['value', 'size', 'quality', 'volatility', 'return']
    factor_names = ['Value', 'Size', 'Quality', 'Volatility', 'Return']
    
    # Filter to only available columns
    available_cols = [col for col in factor_cols if col in df.columns]
    available_names = [factor_names[i] for i, col in enumerate(factor_cols) if col in df.columns]
    
    stats_data = []
    for col, name in zip(available_cols, available_names):
        stats_data.append([
            name,
            f"{df[col].mean():.4f}",
            f"{df[col].std():.4f}",
            f"{df[col].skew():.3f}",
            f"{df[col].kurtosis():.3f}"
        ])
    
    # Add correlation with returns row
    stats_data.append(['', '', '', '', ''])  # Separator
    stats_data.append(['Correlation with Returns:', '', '', '', ''])
    
    # Exclude return itself from correlation analysis
    factor_cols_no_return = [col for col in available_cols if col != 'return']
    factor_names_no_return = [available_names[i] for i, col in enumerate(available_cols) if col != 'return']
    
    for col, name in zip(factor_cols_no_return, factor_names_no_return):
        corr = df[col].corr(df['return'])
        stats_data.append([f'  {name}', f'{corr:.4f}', '', '', ''])
    
    # Create table
    headers = ['Variable', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis']
    
    # Table styling
    table = ax.table(cellText=stats_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight correlation section
    start_corr_row = len(factor_names) + 2
    for i in range(start_corr_row, len(stats_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('#F0F8FF')
    
    # Add panel label
    ax.text(-0.1, 0.95, '(d)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Add data generation info
    n_stocks = metrics.get('synthetic_n_stocks', 100)
    n_months = metrics.get('synthetic_n_months', 48)
    n_obs = metrics.get('synthetic_n_observations_total', 4800)
    seed_val = metrics.get('synthetic_seed_value', 42)
    
    info_text = f"Data: {n_stocks} stocks × {n_months} months = {n_obs:,} observations\nSeed: {seed_val}"
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='bottom', style='italic')

def main():
    """Generate synthetic data correlation analysis figure."""
    print("Generating Synthetic Data Correlation Analysis Figure")
    
    # Load metrics
    metrics = load_metrics()
    seed_val = metrics.get('synthetic_seed_value', 42)
    
    # Generate synthetic data
    print(f"📊 Generating synthetic data with seed {seed_val}...")
    df = generate_synthetic_data(random_seed=seed_val)
    
    print(f"✅ Generated data: {len(df):,} observations")
    print(f"   Stocks: {df['stock_id'].nunique()}")
    print(f"   Time periods: {df['month'].nunique()}")
    
    # Create the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('')  # No main title for LaTeX integration
    
    # Generate each panel
    print("📈 Creating correlation matrix panel...")
    create_correlation_matrix_panel(ax1, df)
    
    print("📊 Creating effects comparison panel...")
    create_effects_comparison_panel(ax2, df, metrics)
    
    print("🔍 Creating quality scatter panel...")
    create_quality_scatter_panel(ax3, df)
    
    print("📋 Creating summary statistics panel...")
    create_summary_table_panel(ax4, df, metrics)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
    
    # Save the figure
    output_path = Path(__file__).parent.parent / "Synthetic" / "synthetic_data_correlation_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Figure saved: {output_path}")
    print("Synthetic Data Correlation Analysis Complete!")
    
    return output_path

if __name__ == "__main__":
    main() 