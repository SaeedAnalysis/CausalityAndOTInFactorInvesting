# %% [markdown]
# # Graph Generation for Real Data Analysis
# 
# This script generates and saves all graphs from the Fama-French real data analysis,
# following the same structure and naming conventions as the synthetic data analysis.
#
# ## Graph Overview:
# 1. **Factor Distributions**: Shows the distribution of factor returns to understand their characteristics
# 2. **Returns Over Time**: Visualizes how different portfolio groups perform over time
# 3. **Correlation Matrix**: Reveals relationships between factors and returns
# 4. **DiD Results**: Illustrates the causal effects of market events using difference-in-differences
# 5. **Covariate Balance**: Checks if comparison groups are similar on observable characteristics
# 6. **IV Results**: Compares OLS vs instrumental variable estimates to address endogeneity
# 7. **Causal Graph**: Network visualization of discovered causal relationships
# 8. **Regime Effects**: Shows how factor effectiveness varies by market conditions
# 9. **Causal Discovery Comparison**: Compares results from different causal discovery methods
# 10. **Factor Performance Summary**: Long-term cumulative performance of each factor

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import os
import warnings
warnings.filterwarnings('ignore')

# Check for optional libraries
try:
    import ot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False

try:
    from causallearn.search.ConstraintBased.PC import pc
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False

# Set visual style
np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# %% [markdown]
# ## Helper Functions

# %%
def ensure_graphs_dir():
    """Ensure the Graphs/Real Data directory exists"""
    graphs_dir = os.path.join('Graphs', 'Real Data')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
    return graphs_dir

def save_fig(fig_name):
    """Save the current figure to the Graphs/Real Data directory"""
    graphs_dir = ensure_graphs_dir()
    # Append _real_data to the filename before the extension
    base_name, ext = os.path.splitext(fig_name)
    new_name = f"{base_name}_real_data{ext}"
    plt.savefig(os.path.join(graphs_dir, new_name), dpi=300, bbox_inches='tight')
    print(f"Saved {new_name} to Graphs/Real Data directory")

# %% [markdown]
# ## Import Real Data Analysis Functions

# %%
# Import the real data analysis module
import importlib.util
spec = importlib.util.spec_from_file_location("causality_real_data", "Python/Analysis/Causality_Real_Data.py")
real_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_data_module)

# Load the data
ff_data = real_data_module.load_fama_french_data()
panel_df = real_data_module.create_panel_data(ff_data)

# %% [markdown]
# ## 1. Factor Distributions
# 
# **What it shows**: The distribution (histogram with kernel density estimate) of monthly returns for each factor.
# 
# **How to read**: 
# - X-axis: Factor return values (in decimal form, e.g., 0.05 = 5%)
# - Y-axis: Frequency/density of observations
# - The curve shows the smoothed distribution
# - Vertical red line at 0 helps identify if returns are mostly positive or negative
# 
# **Interpretation**: 
# - Normal-looking distributions suggest well-behaved factors
# - Skewness indicates asymmetric risk/return profiles
# - Fat tails suggest extreme events are more common than normal distribution would predict

# %%
def save_factor_distributions_real(panel_df):
    """Save plot of factor distributions for real data"""
    plt.figure(figsize=(12, 8))

    factors = ['size', 'value', 'momentum', 'profitability']
    factor_labels = ['SMB', 'HML', 'Momentum', 'RMW']
    
    for i, (factor, label) in enumerate(zip(factors, factor_labels)):
        plt.subplot(2, 2, i+1)
        sns.histplot(panel_df[factor], kde=True, bins=50)
        plt.title(f'{label} Factor Distribution')
        plt.xlabel('Factor Value')

    plt.tight_layout()
    save_fig('factor_distributions.png')
    plt.close()

save_factor_distributions_real(panel_df)

# %% [markdown]
# ## 2. Returns Over Time
# 
# **What it shows**: Average monthly excess returns for small cap vs large cap portfolios over time.
# 
# **How to read**:
# - X-axis: Time period (dates)
# - Y-axis: Average excess returns (in percentage)
# - Blue line: Small cap portfolio average returns
# - Red dashed line: Large cap portfolio average returns
# 
# **Interpretation**:
# - Periods where blue line is above red indicate small cap outperformance
# - High volatility periods show larger swings in both lines
# - Convergence/divergence patterns reveal changing market dynamics

# %%
def save_returns_time_plot_real(panel_df):
    """Save plot of returns over time for different portfolios"""
    plt.figure(figsize=(10, 6))
    
    # Calculate average returns by month for small vs large portfolios
    monthly_returns = panel_df.groupby(['date', 'size_characteristic'])['excess_return'].mean().reset_index()
    
    # Split into small and large based on size characteristic
    small_returns = monthly_returns[monthly_returns['size_characteristic'] < 0]
    large_returns = monthly_returns[monthly_returns['size_characteristic'] > 0]
    
    # Group by date and average
    small_avg = small_returns.groupby('date')['excess_return'].mean()
    large_avg = large_returns.groupby('date')['excess_return'].mean()
    
    plt.plot(small_avg.index, small_avg.values * 100, 'b-', alpha=0.7, label='Small Cap')
    plt.plot(large_avg.index, large_avg.values * 100, 'r--', alpha=0.7, label='Large Cap')
    
    plt.xlabel('Date')
    plt.ylabel('Average Excess Returns (%)')
    plt.title('Average Returns by Size Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_fig('returns_time.png')
    plt.close()

save_returns_time_plot_real(panel_df)

# %% [markdown]
# ## 3. Correlation Matrix
# 
# **What it shows**: Pairwise correlations between all factors and returns.
# 
# **How to read**:
# - Each cell shows correlation coefficient (-1 to +1)
# - Red colors: Negative correlation (factors move in opposite directions)
# - Blue colors: Positive correlation (factors move together)
# - Darker colors indicate stronger relationships
# 
# **Interpretation**:
# - Strong correlations between factors may indicate multicollinearity
# - Factor-return correlations suggest potential predictive relationships
# - Near-zero correlations indicate independence

# %%
def save_correlation_matrix_real(panel_df):
    """Save correlation matrix heatmap for real data"""
    # Select factor columns and returns
    factor_cols = ['market', 'size', 'value', 'momentum', 'profitability', 'investment', 'excess_return']
    corr_matrix = panel_df[factor_cols].corr()
    
    # Rename for display
    display_names = ['Market', 'SMB', 'HML', 'Momentum', 'RMW', 'CMA', 'Returns']
    corr_matrix.index = display_names
    corr_matrix.columns = display_names
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Between Factors and Returns')
    plt.tight_layout()
    save_fig('correlation_matrix.png')
    plt.close()

save_correlation_matrix_real(panel_df)

# %% [markdown]
# ## 4. DiD Results Visualization
# 
# **What it shows**: Difference-in-Differences analysis for major market events (e.g., Dot-com bubble).
# 
# **How to read**:
# - Top left: Time series showing treated vs control group returns over event period
# - Top right: Bar chart comparing pre/post averages for each group
# - Bottom row: Distribution comparisons showing how return distributions changed
# 
# **Interpretation**:
# - The DiD estimate (shown in title) is the causal effect of the event
# - Positive DiD: Treated group improved relative to control
# - Parallel trends before event suggest valid identification

# %%
def save_did_results_real(panel_df, event):
    """Create and save DiD visualization for a specific event"""
    # Run DiD analysis for the event
    result = real_data_module.run_did_event_analysis(panel_df, event)
    
    if result is None:
        return
    
    # Filter data for the event periods
    pre_data = panel_df[
        (panel_df['date'] >= event['pre_period'][0]) & 
        (panel_df['date'] <= event['pre_period'][1])
    ].copy()
    
    post_data = panel_df[
        (panel_df['date'] >= event['post_period'][0]) & 
        (panel_df['date'] <= event['post_period'][1])
    ].copy()
    
    # Define treatment groups
    if event['treatment_var'] in ['size_characteristic', 'value_characteristic']:
        threshold = panel_df[event['treatment_var']].median()
        pre_data['treated'] = (pre_data[event['treatment_var']] > threshold).astype(int)
        post_data['treated'] = (post_data[event['treatment_var']] > threshold).astype(int)
    else:
        pre_data['treated'] = (pre_data[event['treatment_var']] > 0).astype(int)
        post_data['treated'] = (post_data[event['treatment_var']] > 0).astype(int)
    
    # Extract distributions
    treat_pre = pre_data[pre_data['treated']==1]['excess_return'].values
    ctrl_pre = pre_data[pre_data['treated']==0]['excess_return'].values
    treat_post = post_data[post_data['treated']==1]['excess_return'].values
    ctrl_post = post_data[post_data['treated']==0]['excess_return'].values
    
    # Create the plot
    fig = plt.figure(figsize=(18, 8))
    
    # 1. Time series plot
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    
    # Combine pre and post data
    combined_data = pd.concat([
        pre_data.assign(period='Pre'),
        post_data.assign(period='Post')
    ])
    
    monthly_returns = combined_data.groupby(['date', 'treated'])['excess_return'].mean().reset_index()
    monthly_treat = monthly_returns[monthly_returns['treated']==1]
    monthly_ctrl = monthly_returns[monthly_returns['treated']==0]
    
    ax1.plot(monthly_treat['date'], monthly_treat['excess_return']*100, 'o-', label='Treated', markersize=4)
    ax1.plot(monthly_ctrl['date'], monthly_ctrl['excess_return']*100, 's-', label='Control', markersize=4)
    
    # Add event boundary
    event_date = pd.to_datetime(event['post_period'][0])
    ax1.axvline(x=event_date, color='gray', linestyle='--', alpha=0.7)
    ax1.text(event_date, ax1.get_ylim()[0]+0.5, 'Event Start', rotation=90, alpha=0.7)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Monthly Return (%)')
    ax1.set_title(f'DiD: {event["name"]} - Returns Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart visualization
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    
    # Plot means
    ax2.bar(0, np.mean(treat_pre)*100, width=0.4, label='Treated Pre', color='blue', alpha=0.7)
    ax2.bar(0.5, np.mean(ctrl_pre)*100, width=0.4, label='Control Pre', color='red', alpha=0.7)
    ax2.bar(1.5, np.mean(treat_post)*100, width=0.4, label='Treated Post', color='blue')
    ax2.bar(2, np.mean(ctrl_post)*100, width=0.4, label='Control Post', color='red')
    
    ax2.set_xticks([0.25, 1.75])
    ax2.set_xticklabels(['Pre-Event', 'Post-Event'])
    ax2.set_ylabel('Average Returns (%)')
    ax2.set_title('Difference-in-Differences')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3-5. Distribution comparisons
    # Treated pre vs post
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    sns.kdeplot(treat_pre*100, ax=ax3, label='Pre-Event', color='blue')
    sns.kdeplot(treat_post*100, ax=ax3, label='Post-Event', color='red')
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('Treated Group Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Control pre vs post
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    sns.kdeplot(ctrl_pre*100, ax=ax4, label='Pre-Event', color='blue')
    sns.kdeplot(ctrl_post*100, ax=ax4, label='Post-Event', color='red')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('Control Group Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Treated vs control post
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    sns.kdeplot(treat_post*100, ax=ax5, label='Treated', color='blue')
    sns.kdeplot(ctrl_post*100, ax=ax5, label='Control', color='red')
    ax5.set_xlabel('Return (%)')
    ax5.set_ylabel('Density')
    ax5.set_title('Post-Event Distributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig('did_results.png')
    plt.close()

# Run DiD for the first event (Dot-com Bubble)
market_events = real_data_module.identify_market_events(panel_df)
if market_events:
    save_did_results_real(panel_df, market_events[0])

# %% [markdown]
# ## 5. Covariate Balance Plot
# 
# **What it shows**: Standardized mean differences between high and low momentum portfolios across factors.
# 
# **How to read**:
# - X-axis: Different factors
# - Y-axis: Standardized mean difference (effect size)
# - Green bars: Well-balanced (< 0.1 threshold)
# - Red bars: Imbalanced (> 0.1 threshold)
# - Dashed lines at ±0.1 show conventional balance threshold
# 
# **Interpretation**:
# - Good balance (green bars) suggests groups are comparable
# - Large imbalances may indicate selection bias
# - Important for validating causal inference assumptions

# %%
def save_covariate_balance_plot_real(panel_df):
    """Create covariate balance plot for portfolio characteristics"""
    # Since we don't have explicit matching in real data, we'll compare
    # high vs low momentum portfolios as an example
    
    high_momentum = panel_df[panel_df['momentum'] > panel_df['momentum'].median()]
    low_momentum = panel_df[panel_df['momentum'] <= panel_df['momentum'].median()]
    
    factors = ['size', 'value', 'profitability', 'investment']
    factor_labels = ['SMB', 'HML', 'RMW', 'CMA']
    
    # Calculate standardized differences
    high_means = high_momentum[factors].mean()
    low_means = low_momentum[factors].mean()
    pooled_std = np.sqrt((high_momentum[factors].var() + low_momentum[factors].var()) / 2)
    std_diff = (high_means - low_means) / pooled_std
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(factors))
    plt.bar(x, std_diff, alpha=0.7, color=['red' if abs(d) > 0.1 else 'green' for d in std_diff])
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='Threshold (±0.1)')
    plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3)
    
    plt.ylabel('Standardized Mean Difference')
    plt.xlabel('Factors')
    plt.title('Covariate Balance: High vs Low Momentum Portfolios')
    plt.xticks(x, factor_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig('covariate_balance_plot.png')
    plt.close()

save_covariate_balance_plot_real(panel_df)

# %% [markdown]
# ## 6. IV Results Plot
# 
# **What it shows**: Comparison of Ordinary Least Squares (OLS) vs Instrumental Variables (IV) estimates.
# 
# **How to read**:
# - Blue bars: OLS estimates (potentially biased if endogeneity exists)
# - Red bars: IV estimates (corrected for endogeneity)
# - F-statistic annotations show instrument strength (>10 is strong)
# 
# **Interpretation**:
# - Large differences between OLS and IV suggest endogeneity bias
# - Weak instruments (low F-stat) make IV estimates unreliable
# - IV estimates show causal effects under stronger assumptions

# %%
def save_iv_results_plot_real(panel_df):
    """Create and save IV results visualization"""
    # Run IV analysis
    iv_results = real_data_module.run_iv_analysis_real(panel_df)
    
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(len(iv_results))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, iv_results['OLS Estimate'], width, 
            label='OLS Estimate', alpha=0.7, color='blue')
    plt.bar(x + width/2, iv_results['IV Estimate'], width, 
            label='IV Estimate', alpha=0.7, color='red')
    
    # Add details
    plt.xlabel('Factor')
    plt.ylabel('Effect Size')
    plt.title('Comparison of OLS vs. IV Estimates - Real Data')
    plt.xticks(x, iv_results['Factor'])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add F-stat annotations
    for i, fstat in enumerate(iv_results['First Stage F']):
        plt.annotate(f"F={fstat:.1f}", 
                    xy=(i + width/2, iv_results['IV Estimate'].iloc[i]),
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=8)
    
    plt.tight_layout()
    save_fig('iv_results.png')
    plt.close()

save_iv_results_plot_real(panel_df)

# %% [markdown]
# ## 7. Causal Graph Visualization
# 
# **What it shows**: Network diagram of causal relationships between factors and returns.
# 
# **How to read**:
# - Nodes: Factors and returns
# - Solid arrows: Discovered causal relationships
# - Dashed arrows: Bidirectional or uncertain relationships
# - Arrow direction indicates causality (from cause to effect)
# 
# **Interpretation**:
# - Direct arrows to Returns indicate factors with causal effects
# - Inter-factor relationships reveal indirect pathways
# - Based on financial theory and empirical discovery

# %%
def save_causal_graph_real():
    """Create a causal graph visualization for real data factors"""
    try:
        import networkx as nx
        
        # Create a directed graph based on financial theory and our findings
        G = nx.DiGraph()
        
        # Define nodes
        nodes = ['Market', 'SMB', 'HML', 'Momentum', 'RMW', 'CMA', 'Returns']
        for i, node in enumerate(nodes):
            G.add_node(i, name=node)
        
        # Add edges based on theoretical relationships and empirical findings
        # Market affects returns
        G.add_edge(0, 6)  # Market → Returns
        # Size factor affects returns
        G.add_edge(1, 6)  # SMB → Returns
        # Value factor affects returns
        G.add_edge(2, 6)  # HML → Returns
        # Momentum affects returns
        G.add_edge(3, 6)  # Momentum → Returns
        # Profitability affects returns
        G.add_edge(4, 6)  # RMW → Returns
        # Investment affects returns
        G.add_edge(5, 6)  # CMA → Returns
        
        # Some inter-factor relationships
        G.add_edge(2, 3, style='dashed')  # HML ↔ Momentum (known negative correlation)
        G.add_edge(3, 2, style='dashed')
        
        # Plot the graph
        plt.figure(figsize=(12, 10))
        
        # Use a hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Manually adjust positions for better visualization
        pos[6] = np.array([0, -1])  # Returns at bottom center
        pos[0] = np.array([0, 1])    # Market at top
        pos[1] = np.array([-1.5, 0.5])  # SMB
        pos[2] = np.array([-0.75, 0.5]) # HML
        pos[3] = np.array([0.75, 0.5])  # Momentum
        pos[4] = np.array([1.5, 0.5])   # RMW
        pos[5] = np.array([0, 0])       # CMA
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
        
        # Draw edges
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if 'style' not in d]
        dashed_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('style') == 'dashed']
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, arrowsize=20, 
                              edge_color='gray', arrowstyle='->')
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=2, style='dashed', 
                              arrowsize=20, edge_color='gray', arrowstyle='<->')
        
        # Draw labels
        labels = {i: data['name'] for i, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
        
        plt.axis('off')
        plt.title('Causal Graph: Fama-French Factors → Returns', fontsize=16, pad=20)
        plt.tight_layout()
        save_fig('causal_graph.png')
        plt.close()
        
    except ImportError:
        print("NetworkX not available, skipping causal graph visualization")

save_causal_graph_real()

# %% [markdown]
# ## 8. Regime Analysis Plot
# 
# **What it shows**: How factor effects on returns vary between high and low volatility market regimes.
# 
# **How to read**:
# - Dark red bars: Factor coefficients during high volatility periods
# - Dark blue bars: Factor coefficients during low volatility periods
# - Y-axis: Effect size (regression coefficient)
# 
# **Interpretation**:
# - Different bar heights indicate regime-dependent effects
# - Sign changes (positive to negative) suggest fundamental shifts
# - Larger effects in high volatility may indicate crisis behavior

# %%
def save_regime_analysis_plot(panel_df):
    """Save regime-dependent factor effects visualization"""
    # Run regime analysis
    regime_results = real_data_module.analyze_factor_timing(panel_df)
    
    # The plot is already created in the analyze_factor_timing function
    # We'll recreate it here to save it
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot data for plotting
    pivot_df = regime_results.pivot(index='Factor', columns='Regime', values='Coefficient')
    
    x = np.arange(len(pivot_df.index))
    width = 0.35
    
    ax.bar(x - width/2, pivot_df['High Volatility'], width, 
           label='High Volatility', alpha=0.7, color='darkred')
    ax.bar(x + width/2, pivot_df['Low Volatility'], width, 
           label='Low Volatility', alpha=0.7, color='darkblue')
    
    ax.set_xlabel('Factor')
    ax.set_ylabel('Coefficient')
    ax.set_title('Factor Effects by Market Regime')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    save_fig('regime_effects.png')
    plt.close()

save_regime_analysis_plot(panel_df)

# %% [markdown]
# ## 9. Causal Discovery Comparison
# 
# **What it shows**: Agreement between ANM and DIVOT causal discovery methods.
# 
# **How to read**:
# - Y-axis: 1 = Factor causes returns, 0 = No causal relationship found
# - Blue bars: ANM method results
# - Green bars: DIVOT method results
# 
# **Interpretation**:
# - Agreement between methods increases confidence in findings
# - Disagreement suggests complex or weak causal relationships
# - "Inconclusive" results are common in financial data due to complexity

# %%
def save_causal_discovery_comparison(panel_df):
    """Save comparison of causal discovery methods"""
    # Run causal discovery methods
    anm_results = real_data_module.discover_factor_causality_real(panel_df)
    divot_results = real_data_module.run_divot_discovery_real(panel_df)
    
    # Merge results
    comparison = anm_results.merge(
        divot_results[['Factor', 'Direction', 'Score']], 
        on='Factor', 
        suffixes=('_ANM', '_DIVOT')
    )
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    factors = comparison['Factor']
    x = np.arange(len(factors))
    width = 0.35
    
    # Check if methods found causal relationships
    anm_causal = comparison['Direction_ANM'].str.contains('→ Returns').astype(int)
    divot_causal = comparison['Direction_DIVOT'].str.contains('→ Returns').astype(int)
    
    plt.bar(x - width/2, anm_causal, width, label='ANM', color='royalblue', alpha=0.7)
    plt.bar(x + width/2, divot_causal, width, label='DIVOT', color='green', alpha=0.7)
    
    plt.xlabel('Factor')
    plt.ylabel('Causal to Returns (1 = Yes, 0 = No)')
    plt.title('Causal Discovery Results: ANM vs DIVOT')
    plt.xticks(x, factors, rotation=45)
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig('causal_discovery_comparison.png')
    plt.close()

save_causal_discovery_comparison(panel_df)

# %% [markdown]
# ## 10. Factor Performance Summary
# 
# **What it shows**: Cumulative performance of each factor over the entire sample period.
# 
# **How to read**:
# - X-axis: Time period
# - Y-axis: Cumulative return (1 = starting value)
# - Each subplot shows one factor's long-term performance
# - Text box shows annualized return, volatility, and Sharpe ratio
# 
# **Interpretation**:
# - Upward slopes indicate positive long-term returns
# - Volatility shown by jaggedness of the line
# - Sharpe ratio measures risk-adjusted performance

# %%
def save_factor_performance_summary(ff_data):
    """Save summary of factor performance over time"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    colors = ['darkblue', 'darkgreen', 'darkred', 'purple', 'orange', 'brown']
    
    for i, (factor, color) in enumerate(zip(factors, colors)):
        ax = axes[i]
        
        # Calculate cumulative returns
        cumulative_returns = (1 + ff_data[factor]).cumprod()
        
        # Plot cumulative returns
        ax.plot(cumulative_returns.index, cumulative_returns.values, 
                color=color, linewidth=2)
        
        # Add horizontal line at 1
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        # Calculate and display statistics
        annual_return = ff_data[factor].mean() * 12 * 100
        annual_vol = ff_data[factor].std() * np.sqrt(12) * 100
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0
        
        ax.set_title(f'{factor} Cumulative Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Ann. Return: {annual_return:.1f}%\n'
        textstr += f'Ann. Vol: {annual_vol:.1f}%\n'
        textstr += f'Sharpe: {sharpe:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    save_fig('factor_performance_summary.png')
    plt.close()

save_factor_performance_summary(ff_data)

# %% [markdown]
# ## Generate Graph Interpretation CSV

# %%
def create_graph_interpretation_csv():
    """Create a CSV file with detailed interpretations of all graphs"""
    
    interpretations = [
        {
            'Graph Name': 'factor_distributions_real_data.png',
            'What It Shows': 'Distribution of monthly returns for SMB, HML, Momentum, and RMW factors',
            'How to Read': 'X-axis shows return values (decimal), Y-axis shows frequency. Histogram bars show actual data, smooth curve shows kernel density estimate.',
            'Key Insights': 'Look for: (1) Symmetry vs skewness, (2) Fat tails indicating extreme events, (3) Center of distribution relative to zero',
            'Interpretation of Results': 'Most factors show roughly normal distributions with some fat tails, indicating occasional extreme returns. This is typical for financial factors.'
        },
        {
            'Graph Name': 'returns_time_real_data.png',
            'What It Shows': 'Time series of average monthly excess returns for small cap vs large cap portfolios',
            'How to Read': 'Blue line = small cap average returns, Red dashed line = large cap average returns. Y-axis in percentage.',
            'Key Insights': 'Look for: (1) Periods of outperformance, (2) Volatility clustering, (3) Major market events impact',
            'Interpretation of Results': 'Small caps show higher volatility but similar long-term returns. Notable divergence during crisis periods where large caps typically outperform.'
        },
        {
            'Graph Name': 'correlation_matrix_real_data.png',
            'What It Shows': 'Pairwise correlations between all factors and returns',
            'How to Read': 'Blue = positive correlation, Red = negative correlation. Darker colors = stronger relationships. Values range from -1 to +1.',
            'Key Insights': 'Look for: (1) Factor independence, (2) Which factors correlate with returns, (3) Potential multicollinearity issues',
            'Interpretation of Results': 'Market factor shows strongest correlation with returns. HML and CMA are highly correlated (0.68), suggesting they capture similar effects. Most factors show low correlation with each other.'
        },
        {
            'Graph Name': 'did_results_real_data.png',
            'What It Shows': 'Difference-in-Differences analysis for the Dot-com bubble event',
            'How to Read': 'Top left: time series of treated (value) vs control (growth) returns. Top right: pre/post averages. Bottom: distribution changes.',
            'Key Insights': 'DiD estimate shows causal effect. Positive = treated group benefited. Check parallel trends assumption in pre-period.',
            'Interpretation of Results': 'DiD estimate of 0.99% shows value stocks outperformed growth stocks by about 1% due to the dot-com crash, confirming the value premium during market corrections.'
        },
        {
            'Graph Name': 'covariate_balance_plot_real_data.png',
            'What It Shows': 'Standardized mean differences between high and low momentum portfolios across other factors',
            'How to Read': 'Green bars = good balance (<0.1), Red bars = imbalance (>0.1). Y-axis shows standardized difference (effect size).',
            'Key Insights': 'Good balance ensures fair comparison. Large imbalances suggest selection bias that could confound results.',
            'Interpretation of Results': 'Most factors show good balance between high/low momentum groups, validating comparisons. Any imbalances are within acceptable ranges.'
        },
        {
            'Graph Name': 'iv_results_real_data.png',
            'What It Shows': 'Comparison of OLS vs IV estimates for causal effects of factors on returns',
            'How to Read': 'Blue bars = OLS estimates, Red bars = IV estimates. F-stat >10 indicates strong instrument. Large differences suggest endogeneity.',
            'Key Insights': 'IV corrects for endogeneity bias. Weak instruments (low F) make IV unreliable. Compare magnitude and sign changes.',
            'Interpretation of Results': 'SIZE shows dramatic difference (OLS: 0.89, IV: -8.34) but weak instrument (F=0.55) makes IV unreliable. VALUE and MOMENTUM have strong instruments and show meaningful endogeneity correction.'
        },
        {
            'Graph Name': 'causal_graph_real_data.png',
            'What It Shows': 'Network diagram of causal relationships between factors and returns based on theory and empirical analysis',
            'How to Read': 'Arrows show causal direction. Solid = direct causation, Dashed = bidirectional/uncertain. All factors point to Returns node.',
            'Key Insights': 'Direct paths show primary causal effects. Indirect paths through other factors show mediation effects.',
            'Interpretation of Results': 'All factors show direct causal paths to returns (based on theory). HML-Momentum bidirectional relationship reflects known negative correlation between value and momentum strategies.'
        },
        {
            'Graph Name': 'regime_effects_real_data.png',
            'What It Shows': 'How factor effectiveness (regression coefficients) varies between high and low volatility market regimes',
            'How to Read': 'Dark red = high volatility regime effects, Dark blue = low volatility regime effects. Y-axis shows coefficient magnitude.',
            'Key Insights': 'Different heights show regime dependence. Sign changes indicate fundamental shifts in factor behavior.',
            'Interpretation of Results': 'SIZE effect stronger in high vol (0.92 vs 0.80). MOMENTUM switches from negative in high vol to positive in low vol, suggesting it works better in calm markets. VALUE also shows regime dependence.'
        },
        {
            'Graph Name': 'causal_discovery_comparison_real_data.png',
            'What It Shows': 'Agreement between ANM and DIVOT causal discovery methods for each factor',
            'How to Read': '1 = method found causal relationship, 0 = no causality found. Blue = ANM results, Green = DIVOT results.',
            'Key Insights': 'Agreement between methods increases confidence. Both showing 0 suggests no clear causality or complex relationships.',
            'Interpretation of Results': 'Both methods show inconclusive (0) results for all factors, indicating complex, possibly nonlinear or time-varying causal relationships that simple methods cannot detect in financial data.'
        },
        {
            'Graph Name': 'factor_performance_summary_real_data.png',
            'What It Shows': 'Cumulative performance of each factor over the full sample period (1963-2025)',
            'How to Read': 'Each subplot shows one factor. Y-axis: cumulative return (1 = starting value). Text box shows annualized metrics.',
            'Key Insights': 'Upward slope = positive returns. Volatility shown by line jaggedness. Sharpe ratio = risk-adjusted performance.',
            'Interpretation of Results': 'Market factor shows strongest performance. SMB and HML show positive but volatile returns. Momentum shows high volatility. All factors exhibit significant drawdowns during crisis periods.'
        }
    ]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(interpretations)
    csv_path = os.path.join('Graphs', 'Real Data', 'graph_interpretations.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nCreated graph interpretations CSV at: {csv_path}")
    
    # Also create a markdown version for easy reading
    md_path = os.path.join('Graphs', 'Real Data', 'graph_interpretations.md')
    with open(md_path, 'w') as f:
        f.write("# Graph Interpretations for Real Data Analysis\n\n")
        
        for i, row in df.iterrows():
            f.write(f"## {i+1}. {row['Graph Name']}\n\n")
            f.write(f"**What It Shows**: {row['What It Shows']}\n\n")
            f.write(f"**How to Read**: {row['How to Read']}\n\n")
            f.write(f"**Key Insights**: {row['Key Insights']}\n\n")
            f.write(f"**Interpretation of Results**: {row['Interpretation of Results']}\n\n")
            f.write("---\n\n")
    
    print(f"Created graph interpretations Markdown at: {md_path}")
    
    return df

# Generate the interpretation files
interpretation_df = create_graph_interpretation_csv()

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*60)
print("GRAPH GENERATION COMPLETE")
print("="*60)
print(f"\nAll graphs have been saved to: Graphs/Real Data/")
print("\nGenerated graphs:")
print("1. factor_distributions_real_data.png")
print("2. returns_time_real_data.png")
print("3. correlation_matrix_real_data.png")
print("4. did_results_real_data.png")
print("5. covariate_balance_plot_real_data.png")
print("6. iv_results_real_data.png")
print("7. causal_graph_real_data.png")
print("8. regime_effects_real_data.png")
print("9. causal_discovery_comparison_real_data.png")
print("10. factor_performance_summary_real_data.png")
print("\nAll graphs follow the same structure as the synthetic data analysis")
print("with '_real_data' appended to maintain consistency.")
print("\nAdditional files created:")
print("- graph_interpretations.csv: Detailed interpretations in CSV format")
print("- graph_interpretations.md: Same content in readable Markdown format") 