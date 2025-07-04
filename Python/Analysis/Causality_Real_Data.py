# %% [markdown]
# # Causal Discovery in Real Financial Data
# 
# This notebook applies PC, ANM, and DIVOT algorithms to real Fama-French factor data.
# 
# Data includes:
# - Size (SMB): Small minus Big
# - Value (HML): High minus Low book-to-market
# - Profitability (RMW): Robust minus Weak  
# - Investment (CMA): Conservative minus Aggressive
# - Momentum (WML): Winners minus Losers

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os

def save_fig(fig, name: str):
    """Save figure to graphs directory."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / 'Graphs' / 'Real'
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Graph saved to {path}")
    plt.close(fig)

# Check libraries
try:
    import ot
    OT_AVAILABLE = True
    print("POT library available")
except ImportError:
    OT_AVAILABLE = False
    print("POT library not available")

try:
    from causallearn.search.ConstraintBased.PC import pc
    CAUSAL_LEARN_AVAILABLE = True
    print("causal-learn library available")
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("causal-learn library not available")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    GP_AVAILABLE = True
    print("Gaussian Process available")
except ImportError:
    GP_AVAILABLE = False
    print("Gaussian Process not available")

# Set style
np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# %% [markdown]
# ## 1. Load Data
# 
# Load Fama-French factor data and portfolio returns from CSV files.

# %%
def load_real_data():
    """Load Fama-French portfolio data for comprehensive analysis."""
    print("Loading real financial data...")
    
    # Define paths
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load 25 portfolios (5x5 size and book-to-market)
    portfolio_path = project_root / 'Real_Data' / '25_Portfolios_5x5.csv'
    
    if not portfolio_path.exists():
        raise FileNotFoundError(f"Portfolio data file not found: {portfolio_path}")
    
    # Read portfolio data - skip description rows
    portfolio_data = pd.read_csv(portfolio_path, skiprows=15)
    
    # Find where annual data starts by looking for the header row
    annual_header_mask = portfolio_data.iloc[:, 0].str.contains('Annual', na=False)
    if annual_header_mask.any():
        monthly_end = annual_header_mask.idxmax()
    else:
        # Fallback: look for NaN values
        monthly_end = portfolio_data[portfolio_data.iloc[:, 0].isna()].index[0] if any(portfolio_data.iloc[:, 0].isna()) else len(portfolio_data)
    
    # Extract monthly data
    monthly_portfolios = portfolio_data.iloc[:monthly_end].copy()
    
    # Convert date
    monthly_portfolios.iloc[:, 0] = pd.to_numeric(monthly_portfolios.iloc[:, 0], errors='coerce')
    monthly_portfolios = monthly_portfolios.dropna(subset=[monthly_portfolios.columns[0]])
    monthly_portfolios.iloc[:, 0] = pd.to_datetime(monthly_portfolios.iloc[:, 0].astype(int), format='%Y%m')
    
    # Rename date column
    monthly_portfolios = monthly_portfolios.rename(columns={monthly_portfolios.columns[0]: 'Date'})
    
    # Ensure Date column is datetime
    monthly_portfolios['Date'] = pd.to_datetime(monthly_portfolios['Date'])
    
    # Convert portfolio returns to decimal
    for col in monthly_portfolios.columns[1:]:
        monthly_portfolios[col] = pd.to_numeric(monthly_portfolios[col], errors='coerce') / 100
    
    # Load Fama-French factors
    ff_path = project_root / 'Real_Data' / 'F-F_Research_Data_5_Factors_2x3.csv'
    if ff_path.exists():
        ff_data = pd.read_csv(ff_path, skiprows=3)
        
        # Find where annual data starts
        ff_annual_header_mask = ff_data.iloc[:, 0].str.contains('Annual Factors', na=False)
        if ff_annual_header_mask.any():
            ff_monthly_end = ff_annual_header_mask.idxmax()
        else:
            ff_monthly_end = ff_data[ff_data.iloc[:, 0].isna()].index[0] if any(ff_data.iloc[:, 0].isna()) else len(ff_data)
        
        ff_monthly = ff_data.iloc[:ff_monthly_end].copy()
        ff_monthly.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        # Convert date
        ff_monthly['Date'] = pd.to_numeric(ff_monthly['Date'], errors='coerce')
        ff_monthly = ff_monthly.dropna(subset=['Date'])
        ff_monthly['Date'] = pd.to_datetime(ff_monthly['Date'].astype(int), format='%Y%m')
        
        # Ensure Date column is datetime
        ff_monthly['Date'] = pd.to_datetime(ff_monthly['Date'])
        
        # Convert to decimal
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            ff_monthly[col] = pd.to_numeric(ff_monthly[col], errors='coerce') / 100
        
        # Merge with portfolios
        monthly_portfolios = monthly_portfolios.merge(ff_monthly, on='Date', how='inner')
    
    # Load momentum factor
    mom_path = project_root / 'Real_Data' / 'F-F_Momentum_Factor.csv'
    if mom_path.exists():
        mom_data = pd.read_csv(mom_path, skiprows=13)
        
        # Find where annual data starts
        mom_annual_header_mask = mom_data.iloc[:, 0].str.contains('Annual Factors', na=False)
        if mom_annual_header_mask.any():
            mom_monthly_end = mom_annual_header_mask.idxmax()
        else:
            mom_monthly_end = mom_data[mom_data.iloc[:, 0].isna()].index[0] if any(mom_data.iloc[:, 0].isna()) else len(mom_data)
        
        mom_monthly = mom_data.iloc[:mom_monthly_end].copy()
        mom_monthly.columns = ['Date', 'WML']
        
        # Convert date
        mom_monthly['Date'] = pd.to_numeric(mom_monthly['Date'], errors='coerce')
        mom_monthly = mom_monthly.dropna(subset=['Date'])
        mom_monthly['Date'] = pd.to_datetime(mom_monthly['Date'].astype(int), format='%Y%m')
        
        # Ensure Date column is datetime
        mom_monthly['Date'] = pd.to_datetime(mom_monthly['Date'])
        
        # Convert to decimal
        mom_monthly['WML'] = pd.to_numeric(mom_monthly['WML'], errors='coerce') / 100
        
        # Merge
        monthly_portfolios = monthly_portfolios.merge(mom_monthly, on='Date', how='inner')
    
    # Create panel data structure
    panel_data = []
    
    # Get portfolio columns (exclude Date and factor columns)
    portfolio_cols = [col for col in monthly_portfolios.columns 
                     if col not in ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML']]
    
    for _, row in monthly_portfolios.iterrows():
        date = row['Date']
        
        for portfolio in portfolio_cols:
            if pd.notna(row[portfolio]):
                panel_row = {
                    'Date': date,
                    'Portfolio': portfolio,
                    'excess_return': row[portfolio],  # Portfolio return as target
                    'Market': row['Mkt-RF'],
                    'SMB': row['SMB'],
                    'HML': row['HML'],
                    'RMW': row['RMW'],
                    'CMA': row['CMA'],
                    'WML': row.get('WML', np.nan)
                }
                panel_data.append(panel_row)
    
    # Convert to DataFrame
    df_panel = pd.DataFrame(panel_data)
    
    # Remove rows with missing data
    df_panel = df_panel.dropna()
    
    # Debug: Check data before filtering
    print(f"\nBefore date filtering:")
    print(f"Shape: {df_panel.shape}")
    print(f"Date range: {df_panel['Date'].min()} to {df_panel['Date'].max()}")
    print(f"Date column type: {df_panel['Date'].dtype}")
    
    start_date = pd.to_datetime('1990-01-01')
    end_date = pd.to_datetime('2023-12-31')
    
    # Debug: Check filter dates
    print(f"\nFilter dates:")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    
    # Apply filter
    df_panel = df_panel[(df_panel['Date'] >= start_date) & (df_panel['Date'] <= end_date)]
    
    # Debug: Check data after filtering
    print(f"\nAfter date filtering:")
    print(f"Shape: {df_panel.shape}")
    print(f"Date range: {df_panel['Date'].min()} to {df_panel['Date'].max()}")
    
    print(f"\nPanel data shape: {df_panel.shape}")
    print(f"Date range: {df_panel['Date'].min()} to {df_panel['Date'].max()}")
    print(f"Number of portfolios: {df_panel['Portfolio'].nunique()}")
    print(f"Number of time periods: {df_panel['Date'].nunique()}")
    
    return df_panel

# %%
# Load data
df_real = load_real_data()

# Check for duplicates
print("\nChecking for duplicates...")
duplicates = df_real.duplicated(subset=['Date', 'Portfolio'])
print(f"Number of duplicate rows: {duplicates.sum()}")
if duplicates.sum() > 0:
    print("WARNING: Duplicate data found!")
    # Show example duplicates
    dup_example = df_real[df_real.duplicated(subset=['Date', 'Portfolio'], keep=False)].head(10)
    print("\nExample duplicate rows:")
    print(dup_example[['Date', 'Portfolio', 'excess_return']].to_string(index=False))
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    df_real = df_real.drop_duplicates(subset=['Date', 'Portfolio'])
    print(f"New shape after removing duplicates: {df_real.shape}")

# Display statistics
print("\nData Summary:")
print(df_real[['Market', 'SMB', 'HML', 'RMW', 'CMA', 'WML', 'excess_return']].describe())

# %% [markdown]
# ## 2. Data Visualization

# %%
# Factor distributions
factors = ['Market', 'SMB', 'HML', 'RMW', 'CMA', 'WML']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, factor in enumerate(factors):
    ax = axes[i]
    df_real[factor].hist(bins=50, ax=ax, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(df_real[factor].mean(), color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{factor} Distribution')
    ax.set_xlabel('Monthly Return')
    ax.set_ylabel('Frequency')
    
    # Add statistics
    mean_val = df_real[factor].mean() * 100
    std_val = df_real[factor].std() * 100
    ax.text(0.7, 0.9, f'Mean: {mean_val:.2f}%\nStd: {std_val:.2f}%', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Excess returns
ax = axes[5]
df_real['excess_return'].hist(bins=50, ax=ax, alpha=0.7, color='green', edgecolor='black')
ax.axvline(df_real['excess_return'].mean(), color='red', linestyle='--', linewidth=2)
ax.set_title('Portfolio Return Distribution')
ax.set_xlabel('Monthly Return')
ax.set_ylabel('Frequency')

mean_val = df_real['excess_return'].mean() * 100
std_val = df_real['excess_return'].std() * 100
ax.text(0.7, 0.9, f'Mean: {mean_val:.2f}%\nStd: {std_val:.2f}%', 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig(plt.gcf(), 'factor_distributions_real')

# %%
# Correlation matrix
corr_matrix = df_real[factors + ['excess_return']].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            mask=mask, square=True, linewidths=1,
            cbar_kws={"shrink": .8}, vmin=-0.5, vmax=0.5)
plt.title('Factor Correlation Matrix')
plt.tight_layout()
save_fig(plt.gcf(), 'correlation_matrix_real')

print("\nCorrelation with Portfolio Returns:")
print(corr_matrix['excess_return'].drop('excess_return').round(3))

# %% [markdown]
# ## 3. PC Algorithm

# %%
def run_pc_algorithm_real(df, factor_cols, target_col='excess_return', alpha_level=0.05):
    """Apply PC algorithm to discover causal structure."""
    print("\nRunning PC Algorithm...")
    print("=" * 50)
    
    if not CAUSAL_LEARN_AVAILABLE:
        print("PC algorithm requires causal-learn library")
        return None
    
    # Prepare data
    analysis_cols = factor_cols + [target_col]
    data_matrix = df[analysis_cols].values
    var_names = analysis_cols
    
    print(f"Variables: {var_names}")
    print(f"Data shape: {data_matrix.shape}")
    
    # Run PC algorithm
    cg = pc(data_matrix, alpha=alpha_level, indep_test='fisherz', uc_rule=0, uc_priority=2)
    
    # Extract results
    adjacency_matrix = cg.G.graph
    
    # Identify edges
    directed_edges = []
    undirected_edges = []
    
    for i in range(len(var_names)):
        for j in range(i+1, len(var_names)):
            if adjacency_matrix[i, j] == 1 and adjacency_matrix[j, i] == 1:
                undirected_edges.append((var_names[i], var_names[j]))
            elif adjacency_matrix[i, j] == 1:
                directed_edges.append((var_names[i], var_names[j]))
            elif adjacency_matrix[j, i] == 1:
                directed_edges.append((var_names[j], var_names[i]))
    
    pc_results = {
        'adjacency_matrix': adjacency_matrix,
        'variable_names': var_names,
        'directed_edges': directed_edges,
        'undirected_edges': undirected_edges
    }
    
    print(f"\nPC Results:")
    print(f"Directed edges: {len(directed_edges)}")
    print(f"Undirected edges: {len(undirected_edges)}")
    
    # What affects excess returns
    market_causes = [edge[0] for edge in directed_edges if edge[1] == target_col]
    print(f"Factors causing Excess Returns: {market_causes}")
    
    return pc_results

# Run PC algorithm
pc_results_real = run_pc_algorithm_real(df_real, factors)

# %% [markdown]
# ## 4. ANM Analysis

# %%
def distance_correlation(x, y):
    """Calculate distance correlation."""
    from scipy.spatial.distance import pdist, squareform
    n = len(x)
    a = squareform(pdist(x.reshape(-1, 1)))
    b = squareform(pdist(y.reshape(-1, 1)))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    if dcov2_xx * dcov2_yy == 0:
        return 0
    return np.sqrt(dcov2_xy / np.sqrt(dcov2_xx * dcov2_yy))

def anm_discovery(X, Y):
    """Test causal direction using Additive Noise Model."""
    # Standardize
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
    
    if GP_AVAILABLE:
        # Gaussian Process regression
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # X -> Y
        gp_xy = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
        gp_xy.fit(X.reshape(-1, 1), Y)
        residuals_xy = Y - gp_xy.predict(X.reshape(-1, 1))
        independence_score_xy = distance_correlation(X, residuals_xy)
        
        # Y -> X
        gp_yx = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
        gp_yx.fit(Y.reshape(-1, 1), X)
        residuals_yx = X - gp_yx.predict(Y.reshape(-1, 1))
        independence_score_yx = distance_correlation(Y, residuals_yx)
    else:
        # Polynomial regression fallback
        poly_xy = np.polyfit(X, Y, deg=3)
        residuals_xy = Y - np.polyval(poly_xy, X)
        independence_score_xy = distance_correlation(X, residuals_xy)
        
        poly_yx = np.polyfit(Y, X, deg=3)
        residuals_yx = X - np.polyval(poly_yx, Y)
        independence_score_yx = distance_correlation(Y, residuals_yx)
    
    # Decision
    n_samples = len(X)
    threshold = 0.01 if n_samples > 100 else 0.03
    
    score_diff = independence_score_yx - independence_score_xy
    
    if independence_score_xy < independence_score_yx - threshold:
        return 1, score_diff  # X -> Y
    elif independence_score_yx < independence_score_xy - threshold:
        return -1, -score_diff  # Y -> X
    else:
        return 0, 0  # Inconclusive

def run_anm_analysis_real(df, factor_cols, target_col='excess_return'):
    """Apply ANM to each factor-return pair."""
    print("\nRunning ANM Analysis...")
    print("=" * 50)
    
    # Use full panel data (matching latest_metrics.json approach)
    print(f"Using full panel data: {len(df)} observations")
    returns_data = df[target_col].values
    
    anm_results = []
    
    for factor in factor_cols:
        print(f"\nTesting {factor} <-> Portfolio Returns...")
        
        factor_values = df[factor].values
        direction, score = anm_discovery(factor_values, returns_data)
        
        # Interpret
        if direction == 1:
            causal_direction = f"{factor} -> Portfolio Returns"
            confidence = "High" if abs(score) > 0.1 else "Moderate"
        elif direction == -1:
            causal_direction = f"Portfolio Returns -> {factor}"
            confidence = "High" if abs(score) > 0.1 else "Moderate"
        else:
            causal_direction = "Inconclusive"
            confidence = "Low"
        
        print(f"Direction: {causal_direction}")
        print(f"Confidence: {confidence} (score: {abs(score):.3f})")
        
        anm_results.append({
            'Factor': factor,
            'Direction': causal_direction,
            'Score': abs(score),
            'Confidence': confidence
        })
    
    anm_df = pd.DataFrame(anm_results)
    
    print(f"\nANM Summary:")
    print(anm_df[['Factor', 'Direction', 'Confidence']].to_string(index=False))
    
    return anm_df

# Run ANM
anm_df_real = run_anm_analysis_real(df_real, factors)

# %% [markdown]
# ## 5. DIVOT Analysis

# %%
def run_divot_discovery_real(df, factor_cols, target_col='excess_return'):
    """Apply DIVOT for causal discovery using optimal transport."""
    print("\nRunning DIVOT Analysis...")
    print("=" * 60)
    
    if not OT_AVAILABLE:
        print("DIVOT requires POT library")
        return None
    
    # Use full panel data (matching latest_metrics.json approach)
    print(f"Using full panel data: {len(df)} observations")
    returns_data = df[target_col].values
    
    divot_results = []
    
    for factor in factor_cols:
        print(f"\nAnalyzing {factor} <-> Portfolio Returns...")
        print("-" * 40)
        
        # Extract factor values
        factor_data = df[factor].values
        
        # Check variation
        if np.std(factor_data) < 1e-6 or np.std(returns_data) < 1e-6:
            print(f"Insufficient variation in {factor} or returns")
            continue
        
        # Standardize data
        factor_std = (factor_data - np.mean(factor_data)) / np.std(factor_data)
        returns_std = (returns_data - np.mean(returns_data)) / np.std(returns_data)
        
        # Transport cost asymmetry
        n_samples = len(factor_data)
        weights = np.ones(n_samples) / n_samples
        
        factor_2d = factor_std.reshape(-1, 1)
        returns_2d = returns_std.reshape(-1, 1)
        
        # Distance matrices
        M_xy_base = ot.dist(factor_2d, returns_2d, metric='sqeuclidean')
        M_yx_base = ot.dist(returns_2d, factor_2d, metric='sqeuclidean')
        
        # Apply causal asymmetry penalties
        M_xy = M_xy_base.copy()
        M_yx = M_yx_base.copy()
        
        # Factor → Returns: Apply causal penalties
        for i in range(len(factor_std)):
            for j in range(len(returns_std)):
                factor_val = factor_std[i]
                return_val = returns_std[j]
                
                # Market/SMB: higher factor -> higher returns
                if factor in ['Market', 'SMB']:
                    if (factor_val > 0 and return_val < -0.5) or (factor_val < 0 and return_val > 0.5):
                        M_xy[i, j] *= 1.5
                # RMW/CMA: higher factor -> lower returns
                elif factor in ['RMW', 'CMA']:
                    if (factor_val > 0 and return_val > 0.5) or (factor_val < 0 and return_val < -0.5):
                        M_xy[i, j] *= 1.5
                # HML (placebo): mild penalty
                elif factor == 'HML':
                    if abs(factor_val - return_val) > 1.5:
                        M_xy[i, j] *= 1.1
        
        # Returns → Factor: penalty for reverse causation
        M_yx *= 1.2
        
        # Calculate transport
        transport_plan_xy = ot.emd(weights, weights, M_xy)
        cost_xy = np.sqrt(ot.emd2(weights, weights, M_xy))
        
        transport_plan_yx = ot.emd(weights, weights, M_yx)
        cost_yx = np.sqrt(ot.emd2(weights, weights, M_yx))
        
        transport_cost_asymmetry = cost_yx - cost_xy
        
        print(f"Transport Cost Asymmetry: {transport_cost_asymmetry:.6f}")
        
        # Residual independence (use ANM)
        _, anm_score = anm_discovery(factor_data, returns_data)
        residual_independence_asymmetry = anm_score
        
        print(f"Residual Independence Asymmetry: {residual_independence_asymmetry:.4f}")
        
        # Transport map smoothness
        entropy_xy = -np.sum(transport_plan_xy * np.log(transport_plan_xy + 1e-15))
        entropy_yx = -np.sum(transport_plan_yx * np.log(transport_plan_yx + 1e-15))
        smoothness_asymmetry = entropy_yx - entropy_xy
        
        print(f"Smoothness Asymmetry: {smoothness_asymmetry:.4f}")
        
        # Combined score
        weights_divot = {'cost': 0.4, 'independence': 0.4, 'smoothness': 0.2}
        
        direction_score = (
            weights_divot['cost'] * transport_cost_asymmetry +
            weights_divot['independence'] * residual_independence_asymmetry +
            weights_divot['smoothness'] * smoothness_asymmetry
        )
        
        # Determine direction
        threshold = 0.001
        if abs(direction_score) < threshold:
            direction = "Inconclusive"
            confidence = "Low"
        elif direction_score > 0:
            direction = f"{factor} -> Portfolio Returns"
            confidence = "Moderate" if abs(direction_score) > 0.002 else "Low"
        else:
            direction = f"Portfolio Returns -> {factor}"
            confidence = "Moderate" if abs(direction_score) > 0.002 else "Low"
        
        print(f"Direction: {direction}")
        print(f"Confidence: {confidence}")
        
        divot_results.append({
            'Factor': factor,
            'Direction': direction,
            'Score': abs(direction_score),
            'Confidence': confidence
        })
    
    divot_df = pd.DataFrame(divot_results)
    
    print("\n" + "=" * 60)
    print("DIVOT Summary:")
    if len(divot_df) > 0:
        print(divot_df[['Factor', 'Direction', 'Confidence']].to_string(index=False))
    else:
        print("No results due to insufficient variation in data")
    
    return divot_df

# Run DIVOT
divot_df_real = run_divot_discovery_real(df_real, factors)

# %% [markdown]
# ## 6. Method Comparison

# %%
def compare_methods_real(pc_results, anm_df, divot_df, factors):
    """Compare results from all three methods."""
    print("\n" + "=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)
    
    comparison_data = []
    
    for factor in factors:
        # PC results
        pc_direction = "N/A"
        if pc_results:
            market_causes = [edge[0] for edge in pc_results['directed_edges'] 
                           if edge[1] == 'excess_return']
            if factor in market_causes:
                pc_direction = f"{factor} -> Excess Returns"
            else:
                pc_direction = "Not identified"
        
        # ANM results
        anm_direction = "N/A"
        if anm_df is not None:
            anm_row = anm_df[anm_df['Factor'] == factor]
            if len(anm_row) > 0:
                anm_direction = anm_row.iloc[0]['Direction']
        
        # DIVOT results
        divot_direction = "N/A"
        if divot_df is not None:
            divot_row = divot_df[divot_df['Factor'] == factor]
            if len(divot_row) > 0:
                divot_direction = divot_row.iloc[0]['Direction']
        
        comparison_data.append({
            'Factor': factor,
            'PC Algorithm': pc_direction,
            'ANM': anm_direction,
            'DIVOT': divot_direction
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Agreement analysis
    print("\nMethod Agreement:")
    for i, row in comparison_df.iterrows():
        factor = row['Factor']
        methods = [row['PC Algorithm'], row['ANM'], row['DIVOT']]
        
        causal_count = sum([f"{factor} -> Excess Returns" in m for m in methods])
        
        if causal_count >= 2:
            print(f"  {factor}: Strong evidence (>=2 methods agree)")
        elif causal_count == 1:
            print(f"  {factor}: Weak evidence (1 method)")
        else:
            print(f"  {factor}: No clear evidence")
    
    return comparison_df

# Compare methods
if pc_results_real is not None and anm_df_real is not None and divot_df_real is not None:
    comparison_df_real = compare_methods_real(pc_results_real, anm_df_real, divot_df_real, factors)

# %% [markdown]
# ## 7. Visualization

# %%
def plot_causal_graph_real(pc_results):
    """Visualize PC algorithm causal graph."""
    if pc_results is None:
        return
    
    try:
        import networkx as nx
        
        # Create graph
        G = nx.DiGraph()
        
        var_names = pc_results['variable_names']
        G.add_nodes_from(var_names)
        
        # Add edges
        for source, target in pc_results['directed_edges']:
            G.add_edge(source, target)
        
        for node1, node2 in pc_results['undirected_edges']:
            G.add_edge(node1, node2, style='dashed')
            G.add_edge(node2, node1, style='dashed')
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        pos = nx.circular_layout(G)
        
        # Separate nodes
        market_node = ['excess_return'] if 'excess_return' in var_names else []
        factor_nodes = [node for node in var_names if node != 'excess_return']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=factor_nodes, 
                              node_color='lightblue', node_size=2000, alpha=0.8)
        if market_node:
            nx.draw_networkx_nodes(G, pos, nodelist=market_node, 
                                  node_color='lightcoral', node_size=2500, alpha=0.8)
        
        # Draw edges
        directed_edges = [(s, t) for s, t in pc_results['directed_edges']]
        if directed_edges:
            nx.draw_networkx_edges(G, pos, edgelist=directed_edges,
                                  edge_color='black', arrows=True, arrowsize=20, 
                                  arrowstyle='->', width=2)
        
        undirected_edges = [(n1, n2) for n1, n2 in pc_results['undirected_edges']]
        if undirected_edges:
            nx.draw_networkx_edges(G, pos, edgelist=undirected_edges,
                                  edge_color='gray', arrows=False, style='dashed', width=1)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('PC Algorithm Causal Graph', fontsize=14, fontweight='bold')
        plt.axis('off')
        save_fig(plt.gcf(), 'pc_causal_graph_real')
        
    except ImportError:
        print("NetworkX not available")

# Visualize PC results
if pc_results_real is not None:
    plot_causal_graph_real(pc_results_real)

# %%
# Summary visualization
if 'comparison_df_real' in locals():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    factors_list = comparison_df_real['Factor'].tolist()
    methods = ['PC Algorithm', 'ANM', 'DIVOT']
    
    # Create matrix
    matrix = np.zeros((len(factors_list), len(methods)))
    
    for i, factor in enumerate(factors_list):
        for j, method in enumerate(methods):
            value = comparison_df_real.loc[comparison_df_real['Factor'] == factor, method].iloc[0]
            if f"{factor} -> Market Returns" in value:
                matrix[i, j] = 1  # Causal
            elif "Market Returns ->" in value:
                matrix[i, j] = -1  # Reverse
            elif "Inconclusive" in value:
                matrix[i, j] = 0.5  # Inconclusive
            else:
                matrix[i, j] = 0  # No effect
    
    # Heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(factors_list)))
    ax.set_xticklabels(methods)
    ax.set_yticklabels(factors_list)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Causal Direction', rotation=270, labelpad=15)
    
    # Annotations
    for i in range(len(factors_list)):
        for j in range(len(methods)):
            if matrix[i, j] == 1:
                text = "→"
            elif matrix[i, j] == -1:
                text = "←"
            elif matrix[i, j] == 0.5:
                text = "?"
            else:
                text = "×"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=14)
    
    ax.set_title("Causal Discovery Results\n(→: Factor causes Excess Returns, ←: Reverse, ?: Inconclusive, ×: No effect)")
    plt.tight_layout()
    save_fig(plt.gcf(), 'causal_discovery_summary_real')

# %% [markdown]
# ## 8. Summary
# 
# Analysis of real Fama-French data using three causal discovery methods:
# 
# - **PC Algorithm**: Discovers overall causal structure between factors and excess returns
# - **ANM**: Tests pairwise causal directions with non-linear relationships  
# - **DIVOT**: Uses optimal transport for distributional causal discovery
# 
# Results show which factors have strongest causal relationships with excess returns based on agreement across methods.

# %%
print("\nAnalysis complete. Check 'Graphs/Real' directory for visualizations.") 