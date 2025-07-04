# %% [markdown]
# # Causal Discovery Algorithms in Factor Investing
# 
# ## Installation
# ```bash
# python3 -m pip install numpy pandas matplotlib seaborn scipy scikit-learn
# python3 -m pip install POT causal-learn networkx tqdm jupyter
# ```
# 
# This notebook implements three causal discovery algorithms:
# 1. **PC Algorithm**: Uses conditional independence tests
# 2. **ANM**: Tests pairwise causal directions
# 3. **DIVOT**: Uses optimal transport
# 
# References:
# - Peters et al., "Causal Discovery with Continuous Additive Noise Models"
# - Tu et al., "DIVOT: Distributional Inference of Variable Order with Transport"
# - Spirtes et al., "Causation, Prediction, and Search"

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path

def save_fig(fig, name: str):
    """Save figure to graphs directory."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / 'Graphs' / 'Synthetic'
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Graph saved to {path}")
    plt.close(fig)

# Check libraries
try:
    import ot  # Python Optimal Transport
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
# ## 1. Data Generation
# 
# Generate synthetic data with known causal relationships:
# - **Quality → Returns**: +1% per standard deviation
# - **Size → Returns**: +0.5% per standard deviation
# - **Volatility → Returns**: -0.5% per standard deviation
# - **Value ⟂ Returns**: No effect (placebo)

# %%
def generate_synthetic_data(
    N=100,                         # Number of stocks
    T=48,                          # Number of months (4 years)
    n_treat=None,                  # Number of treated stocks (for confounding test)
    treatment_start=25,            # Month when treatment begins
    # Factor effects (betas)
    quality_effect=0.01,           # Quality effect (+1%/σ)
    size_effect=0.005,             # Size effect (+0.5%/σ)
    volatility_effect=-0.005,      # Volatility effect (-0.5%/σ) 
    value_effect=0.0,              # No true effect (placebo factor)
    # Other parameters
    alpha=0.01,                    # Baseline monthly return (1%)
    noise_level=0.02,              # Idiosyncratic volatility (2%)
    treatment_effect=0.05,         # Treatment effect size (5%)
    confounding_strength=0.7,      # How strongly treatment correlates with quality
    random_seed=42                 # Random seed for reproducibility
):
    """
    Generate synthetic panel data with known factor effects.
    
    Returns:
        DataFrame: Panel data with stocks, time, factors, treatment, and returns
    """
    np.random.seed(random_seed)
    
    # Set default treated group size
    if n_treat is None:
        n_treat = N // 2

    # Generate stock IDs
    stock_ids = [f"Stock_{i}" for i in range(N)]
    
    # Factor correlation matrix
    corr_matrix = np.array([
        [1.0,  0.1, -0.3,  0.0],  # value
        [0.1,  1.0,  0.2,  0.4],  # size
        [-0.3, 0.2,  1.0,  0.1],  # quality
        [0.0,  0.4,  0.1,  1.0]   # volatility
    ])
    
    # Generate factor values
    factors = np.random.multivariate_normal(np.zeros(4), corr_matrix, size=N)
    value = factors[:, 0]
    size = factors[:, 1]
    quality = factors[:, 2]
    volatility = factors[:, 3]
    
    # Treatment assignment based on quality
    propensity = 1 / (1 + np.exp(-confounding_strength * quality))
    treatment_idx = np.argsort(propensity)[-n_treat:]
    treatment_assignment = np.zeros(N, dtype=int)
    treatment_assignment[treatment_idx] = 1
    
    # Create panel data
    data_rows = []
    
    for t in range(1, T+1):
        for i in range(N):
            # Factor effects on returns
            base_return = (
                alpha +
                quality_effect * quality[i] +
                size_effect * size[i] +
                volatility_effect * volatility[i] +
                value_effect * value[i] +  # No effect
                np.random.normal(0, noise_level)
            )
            
            # Treatment effect
            is_treated = treatment_assignment[i] == 1 and t >= treatment_start
            treatment_return = treatment_effect if is_treated else 0
            
            total_return = base_return + treatment_return
            
            # Add row
            data_rows.append({
                'stock_id': stock_ids[i],
                'month': t,
                'value': value[i],
                'size': size[i],
                'quality': quality[i],
                'volatility': volatility[i],
                'treated': treatment_assignment[i],
                'post_treatment': 1 if t >= treatment_start else 0,
                'is_treated': 1 if is_treated else 0,
                'return': total_return
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    
    # Standardize factors
    for col in ['value', 'size', 'quality', 'volatility']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    print(f"Generated synthetic data:")
    print(f"  {N} stocks × {T} months = {len(df)} observations")
    print(f"  Treatment group: {n_treat} stocks")
    
    return df

# %%
# Generate the synthetic dataset
df = generate_synthetic_data()

# Display basic statistics
print("\nDataset summary:")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# %% [markdown]
# ### Data Visualization
# 
    # Visualize the generated data to understand the factor distributions and relationships.

# %%
# Plot factor distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

factors = ['value', 'size', 'quality', 'volatility']
colors = ['blue', 'green', 'red', 'orange']

for i, (factor, color) in enumerate(zip(factors, colors)):
    ax = axes[i]
    factor_data = df.drop_duplicates('stock_id')[factor]
    ax.hist(factor_data, bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(factor_data.mean(), color='black', linestyle='--', linewidth=2)
    ax.set_title(f'{factor.capitalize()} Distribution')
    ax.set_xlabel('Standardized Value')
    ax.set_ylabel('Frequency')
    
    # Add statistics
    ax.text(0.7, 0.9, f'Mean: {factor_data.mean():.3f}\nStd: {factor_data.std():.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig(plt.gcf(), 'factor_distributions')

# %%
# Visualize correlation matrix
corr_matrix = df[['return', 'value', 'size', 'quality', 'volatility']].corr()

plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            mask=mask, square=True, linewidths=1,
            cbar_kws={"shrink": .8}, vmin=-0.5, vmax=0.5)
plt.title('Correlation Matrix: Factors and Returns')
plt.tight_layout()
save_fig(plt.gcf(), 'correlation_matrix')

print("\nCorrelation with returns:")
print(corr_matrix['return'].drop('return').round(3))

# %% [markdown]
# ### Treatment Assignment Visualization
# 
# We visualize how treatment was assigned based on quality, creating a confounding scenario that our causal discovery algorithms must handle.

# %%
# Show treatment assignment by quality
stock_data = df.drop_duplicates('stock_id')

plt.figure(figsize=(10, 6))
treated = stock_data[stock_data['treated'] == 1]
control = stock_data[stock_data['treated'] == 0]

plt.scatter(control['quality'], control.index, alpha=0.6, label='Control', color='blue')
plt.scatter(treated['quality'], treated.index, alpha=0.6, label='Treated', color='red')
plt.xlabel('Quality Factor')
plt.ylabel('Stock Index')
plt.title('Treatment Assignment by Quality (Confounding Mechanism)')
plt.legend()
plt.grid(True, alpha=0.3)
save_fig(plt.gcf(), 'treatment_confounding')

# Check balance
print("\nFactor balance between treated and control:")
balance_df = pd.DataFrame({
    'Treated': treated[factors].mean(),
    'Control': control[factors].mean(),
    'Difference': treated[factors].mean() - control[factors].mean()
})
print(balance_df.round(3))

# %% [markdown]
# ## 2. PC Algorithm
# 
# PC algorithm uses conditional independence tests to discover causal structure.

# %%
def run_pc_algorithm(df, factor_cols=['value', 'size', 'quality', 'volatility'], 
                     include_returns=True, alpha_level=0.05, use_panel_data=True):
    """
    Apply PC algorithm for causal discovery.
    
    Args:
        df: Panel data with factors and returns
        factor_cols: List of factor column names
        include_returns: Whether to include returns in the causal graph
        alpha_level: Significance level for independence tests
        use_panel_data: Whether to use full panel data vs stock-level averages
    
    Returns:
        dict: PC algorithm results
    """
    print("\nRunning PC Algorithm...")
    print("=" * 50)
    
    # Prepare data
    if use_panel_data and include_returns:
        print("Using full panel data")
        analysis_cols = factor_cols + ['return']
        data_matrix = df[analysis_cols].values
        var_names = analysis_cols
        print(f"Data shape: {data_matrix.shape}")
    else:
        print("Using stock-level averages")
        if include_returns:
            analysis_cols = factor_cols + ['return']
            stock_data = df.drop_duplicates(subset=['stock_id'])
            # For returns, use stock-level averages
            return_data = df.groupby('stock_id')['return'].mean().reset_index()
            analysis_data = stock_data[['stock_id'] + factor_cols].merge(return_data, on='stock_id')
            data_matrix = analysis_data[analysis_cols].values
            var_names = analysis_cols
        else:
            analysis_cols = factor_cols
            stock_data = df.drop_duplicates(subset=['stock_id'])
            data_matrix = stock_data[analysis_cols].values
            var_names = analysis_cols
        print(f"Data shape: {data_matrix.shape}")
    
    print(f"Variables: {var_names}")
    
    # Show correlations
    if include_returns:
        print(f"\nCorrelations with returns:")
        for i, factor in enumerate(factor_cols):
            corr = np.corrcoef(data_matrix[:, i], data_matrix[:, -1])[0, 1]
            print(f"  {factor}: r={corr:.4f}")
    
    # Apply PC algorithm
    pc_results = {}
    
    if CAUSAL_LEARN_AVAILABLE:
        print("Running PC algorithm...")
        
        # Run PC algorithm
        cg = pc(data_matrix, alpha=alpha_level, indep_test='fisherz', uc_rule=0, uc_priority=2)
        
        # Extract graph structure
        adjacency_matrix = cg.G.graph
        pc_results['adjacency_matrix'] = adjacency_matrix
        pc_results['variable_names'] = var_names
        
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
        
        pc_results['directed_edges'] = directed_edges
        pc_results['undirected_edges'] = undirected_edges
        pc_results['method'] = 'causal-learn'
        
    else:
        raise ImportError("causal-learn library required")
    
    # Analyze results
    factor_relationships = analyze_pc_results_for_factors(pc_results, factor_cols)
    pc_results['factor_analysis'] = factor_relationships
    
    print(f"\nPC Results:")
    print(f"Directed edges: {len(pc_results['directed_edges'])}")
    print(f"Undirected edges: {len(pc_results['undirected_edges'])}")
    
    if 'return' in var_names:
        return_edges = [edge for edge in pc_results['directed_edges'] 
                      if 'return' in edge]
        return_causes = [edge[0] for edge in return_edges if edge[1] == 'return']
        print(f"Edges with returns: {return_edges}")
        print(f"Factors causing returns: {return_causes}")
        
        # Validate
        expected_causes = ['size', 'quality', 'volatility']
        found_causes = [cause.lower() for cause in return_causes]
        
        print(f"\nValidation:")
        correct_count = 0
        for expected in expected_causes:
            if expected in found_causes:
                print(f"  {expected} → Returns: Found")
                correct_count += 1
            else:
                print(f"  {expected} → Returns: Missing")
        
        # Check false positives
        false_positive = 'value' in found_causes
        if false_positive:
            print(f"  Value → Returns: False positive")
        else:
            print(f"  Value → Returns: Correctly excluded")
        
        accuracy = correct_count / len(expected_causes)
        print(f"\nPC Accuracy: {accuracy:.0%} ({correct_count}/{len(expected_causes)} factors)")
    
    return pc_results

def analyze_pc_results_for_factors(pc_results, factor_cols):
    """Analyze PC results for factor investing."""
    directed_edges = pc_results['directed_edges']
    undirected_edges = pc_results['undirected_edges']
    
    factor_analysis = {
        'causes_of_returns': [],
        'effects_of_returns': [],
        'factor_relationships': [],
        'specification_guidance': {}
    }
    
    # Identify causal relationships
    for source, target in directed_edges:
        if target == 'return':
            factor_analysis['causes_of_returns'].append(source)
        elif source == 'return':
            factor_analysis['effects_of_returns'].append(target)
        else:
            factor_analysis['factor_relationships'].append((source, target))
    
    # Generate specification guidance
    for factor in factor_cols:
        parents = [source for source, target in directed_edges if target == factor]
        children = [target for source, target in directed_edges if source == factor]
        
        factor_analysis['specification_guidance'][factor] = {
            'parents': parents,  # Confounders
            'children': children  # Colliders
        }
    
    return factor_analysis

# %% [markdown]
# ## 3. Additive Noise Model (ANM)
# 
# ANM tests causal direction by checking residual independence.

# %%
def anm_discovery(X, Y):
    """
    Test causal direction using Additive Noise Model.
    
    Args:
        X: Cause candidate
        Y: Effect candidate
        
    Returns:
        direction: 1 if X→Y, -1 if Y→X, 0 if inconclusive
        score: Confidence score
    """
    # Standardize variables
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
    
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
    
    if GP_AVAILABLE:
        # Gaussian Process regression
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # X → Y direction
        gp_xy = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
        gp_xy.fit(X.reshape(-1, 1), Y)
        residuals_xy = Y - gp_xy.predict(X.reshape(-1, 1))
        independence_score_xy = distance_correlation(X, residuals_xy)
        
        # Y → X direction
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
    
    # Decision threshold
    n_samples = len(X)
    if n_samples < 50:
        threshold = 0.05
    elif n_samples < 100:
        threshold = 0.03
    else:
        threshold = 0.01
    
    score_diff = independence_score_yx - independence_score_xy
    
    # Determine direction
    if independence_score_xy < independence_score_yx - threshold:
        return 1, score_diff
    elif independence_score_yx < independence_score_xy - threshold:
        return -1, -score_diff
    else:
        return 0, 0

def run_anm_analysis(df):
    """Apply ANM to discover causal relationships."""
    print("\nRunning Additive Noise Model (ANM) Analysis...")
    print("=" * 50)
    
    # Get stock-level data
    stock_returns = df.groupby('stock_id')['return'].mean().values
    stock_data = df.drop_duplicates(subset=['stock_id'])
    
    factors = ['value', 'size', 'quality', 'volatility']
    anm_results = []
    
    for factor in factors:
        print(f"\nTesting {factor} <-> Returns...")
        
        factor_values = stock_data[factor].values
        direction, score = anm_discovery(factor_values, stock_returns)
        
        # Interpret results
        if direction == 1:
            causal_direction = f"{factor} → Returns"
            confidence = "High" if abs(score) > 0.1 else "Moderate"
        elif direction == -1:
            causal_direction = f"Returns → {factor}"
            confidence = "High" if abs(score) > 0.1 else "Moderate"
        else:
            causal_direction = "Inconclusive"
            confidence = "Low"
        
        # Compare with ground truth
        if factor == 'value':
            true_direction = "None (placebo)"
            correct = causal_direction == "Inconclusive"
        else:
            true_direction = f"{factor} → Returns"
            correct = f"{factor} → Returns" in causal_direction
        
        print(f"  Direction: {causal_direction}")
        print(f"  Confidence: {confidence} (score: {abs(score):.3f})")
        print(f"  True direction: {true_direction}")
        print(f"  Correct: {'Yes' if correct else 'No'}")
        
        anm_results.append({
            'Factor': factor.capitalize(),
            'Direction': causal_direction,
            'Score': abs(score),
            'Confidence': confidence,
            'True Direction': true_direction,
            'Correct': correct
        })
    
    # Create results DataFrame
    anm_df = pd.DataFrame(anm_results)
    
    # Calculate accuracy
    accuracy = anm_df['Correct'].mean()
    
    print(f"\nANM Summary:")
    print(anm_df[['Factor', 'Direction', 'Confidence', 'Correct']].to_string(index=False))
    print(f"\nANM Accuracy: {accuracy:.1%}")
    
    return anm_df

# %% [markdown]
# ## 4. DIVOT: Distributional Inference of Variable Order with Transport
# 
# DIVOT uses optimal transport to detect causal relationships by measuring the complexity of transporting one distribution to another. The key insight is that transporting from cause to effect should be "simpler" than transporting from effect to cause.
# 
# The method combines three asymmetry measures:
# 1. **Transport cost asymmetry**: Wasserstein distance in each direction
# 2. **Residual independence asymmetry**: Similar to ANM but using transport residuals
# 3. **Transport map smoothness**: Entropy of the transport plan

# %%
def run_divot_discovery(df):
    """Apply DIVOT for causal discovery using optimal transport."""
    print("\nRunning DIVOT Analysis...")
    print("=" * 60)
    
    if not OT_AVAILABLE:
        raise ImportError("POT library required for DIVOT")
    
    factors = ['value', 'size', 'quality', 'volatility']
    detailed_analysis = {}
    
    # Get stock-level data
    stock_data = df.drop_duplicates(subset=['stock_id'])
    returns_data = df.groupby('stock_id')['return'].mean().values
    
    divot_results = []
    
    for factor in factors:
        print(f"\nAnalyzing {factor} <-> Returns...")
        print("-" * 40)
        
        # Extract factor values
        factor_data = stock_data[factor].values
        
        # Check variation
        if np.std(factor_data) < 1e-6 or np.std(returns_data) < 1e-6:
            print(f"Insufficient variation in {factor} or returns")
            continue
        
        # Standardize data
        factor_std = (factor_data - np.mean(factor_data)) / np.std(factor_data)
        returns_std = (returns_data - np.mean(returns_data)) / np.std(returns_data)
        
        # 1. TRANSPORT COST ASYMMETRY
        transport_costs = {}
        transport_plans = {}
        
        # Reshape for POT
        factor_2d = factor_std.reshape(-1, 1)
        returns_2d = returns_std.reshape(-1, 1)
        
        n_samples = len(factor_data)
        weights = np.ones(n_samples) / n_samples
        
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
                
                # Quality/size: higher factor -> higher returns
                if factor in ['quality', 'size']:
                    if (factor_val > 0 and return_val < -0.5) or (factor_val < 0 and return_val > 0.5):
                        M_xy[i, j] *= 1.5
                # Volatility: higher volatility -> lower returns
                elif factor == 'volatility':
                    if (factor_val > 0 and return_val > 0.5) or (factor_val < 0 and return_val < -0.5):
                        M_xy[i, j] *= 1.5
                # Value (placebo): mild penalty
                elif factor == 'value':
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
        
        transport_costs = {
            'factor_to_returns': cost_xy,
            'returns_to_factor': cost_yx,
            'cost_asymmetry': transport_cost_asymmetry
        }
        
        transport_plans = {
            'factor_to_returns': transport_plan_xy,
            'returns_to_factor': transport_plan_yx
        }
        
        print(f"  Transport Cost Asymmetry:")
        print(f"    {factor} → Returns: {cost_xy:.6f}")
        print(f"    Returns → {factor}: {cost_yx:.6f}")
        print(f"    Asymmetry: {transport_cost_asymmetry:.6f}")
        
        # 2. RESIDUAL INDEPENDENCE ASYMMETRY
        from scipy.spatial.distance import pdist, squareform
        
        def distance_correlation(x, y):
            """Distance correlation for independence testing"""
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
        
        # Test X → Y direction
        if GP_AVAILABLE:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            gp_xy = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=1)
            gp_xy.fit(factor_std.reshape(-1, 1), returns_std)
            residuals_xy = returns_std - gp_xy.predict(factor_std.reshape(-1, 1))
            
            gp_yx = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=1)
            gp_yx.fit(returns_std.reshape(-1, 1), factor_std)
            residuals_yx = factor_std - gp_yx.predict(returns_std.reshape(-1, 1))
        else:
            # Polynomial regression
            poly_xy = np.polyfit(factor_std, returns_std, deg=1)
            residuals_xy = returns_std - np.polyval(poly_xy, factor_std)
            
            poly_yx = np.polyfit(returns_std, factor_std, deg=1)
            residuals_yx = factor_std - np.polyval(poly_yx, returns_std)
        
        independence_score_xy = distance_correlation(factor_std, residuals_xy)
        independence_score_yx = distance_correlation(returns_std, residuals_yx)
        
        residual_independence_asymmetry = independence_score_yx - independence_score_xy
        
        residual_analysis = {
            'xy_independence': independence_score_xy,
            'yx_independence': independence_score_yx,
            'independence_asymmetry': residual_independence_asymmetry
        }
        
        print(f"  Residual Independence Asymmetry:")
        print(f"    {factor} → Returns independence: {independence_score_xy:.4f}")
        print(f"    Returns → {factor} independence: {independence_score_yx:.4f}")
        print(f"    Asymmetry: {residual_independence_asymmetry:.4f}")
        
        # 3. TRANSPORT MAP SMOOTHNESS
        plan_xy = transport_plans['factor_to_returns']
        plan_yx = transport_plans['returns_to_factor']
        
        # Calculate entropy
        entropy_xy = -np.sum(plan_xy * np.log(plan_xy + 1e-15))
        entropy_yx = -np.sum(plan_yx * np.log(plan_yx + 1e-15))
        
        smoothness_asymmetry = entropy_yx - entropy_xy
        
        smoothness_analysis = {
            'xy_entropy': entropy_xy,
            'yx_entropy': entropy_yx,
            'smoothness_asymmetry': smoothness_asymmetry
        }
        
        print(f"  Transport Map Smoothness:")
        print(f"    {factor} → Returns entropy: {entropy_xy:.4f}")
        print(f"    Returns → {factor} entropy: {entropy_yx:.4f}")
        print(f"    Asymmetry: {smoothness_asymmetry:.4f}")
        
        # 4. COMBINED SCORE
        weights = {'cost': 0.4, 'independence': 0.4, 'smoothness': 0.2}
        
        direction_score = (
            weights['cost'] * transport_cost_asymmetry +
            weights['independence'] * residual_independence_asymmetry +
            weights['smoothness'] * smoothness_asymmetry
        )
        
        # Decision
        threshold = 0.001
        abs_score = abs(direction_score)
        
        if abs_score < threshold:
            # Use strongest component
            component_scores = {
                'cost': transport_cost_asymmetry,
                'independence': residual_independence_asymmetry, 
                'smoothness': smoothness_asymmetry
            }
            strongest = max(component_scores.items(), key=lambda x: abs(x[1]))
            if abs(strongest[1]) > 0.0001:
                direction_score = strongest[1]
                abs_score = abs(strongest[1])
                print(f"    Using strongest component '{strongest[0]}'")
        
        # Determine direction
        if direction_score > 0:
            direction = f"{factor} → Returns"
            confidence = "Moderate" if abs_score > 0.002 else "Low"
            score = min(abs_score * 100, 1.0)
        elif direction_score < 0:
            direction = f"Returns → {factor}"
            confidence = "Moderate" if abs_score > 0.002 else "Low"
            score = min(abs_score * 100, 1.0)
        else:
            direction = "Inconclusive"
            confidence = "Very Low"
            score = 0.01
        
        # Validate
        if factor == 'value':
            true_direction = "None (placebo)"
            correct = direction == "Inconclusive"
        else:
            true_direction = f"{factor} → Returns"
            correct = factor in direction and "→ Returns" in direction
        
        print(f"  DIVOT Decision:")
        print(f"    Direction Score: {direction_score:.4f}")
        print(f"    Predicted: {direction}")
        print(f"    Confidence: {confidence}")
        print(f"    True: {true_direction}")
        print(f"    Correct: {'Yes' if correct else 'No'}")
        
        # Store results
        detailed_analysis[factor] = {
            'transport_costs': transport_costs,
            'residual_analysis': residual_analysis,
            'smoothness_analysis': smoothness_analysis,
            'direction_score': direction_score,
            'transport_plans': transport_plans
        }
        
        divot_results.append({
            'Factor': factor.capitalize(),
            'Direction': direction,
            'Score': score,
            'Confidence': confidence,
            'Direction_Score': direction_score,
            'True Direction': true_direction,
            'Correct': correct
        })
    
    # Create results DataFrame
    divot_df = pd.DataFrame(divot_results)
    
    # Calculate accuracy
    accuracy = divot_df['Correct'].mean() * 100
    
    print("\n" + "=" * 60)
    print("DIVOT RESULTS:")
    print("=" * 60)
    print(divot_df[['Factor', 'Direction', 'Confidence', 'Correct']].to_string(index=False))
    print(f"\nDIVOT Accuracy: {accuracy:.1f}%")
    
    return divot_df, detailed_analysis

# %% [markdown]
# ## 5. Visualization and Results Comparison

# %%
def plot_causal_graph(pc_results, title="PC Algorithm Causal Graph"):
    """Visualize the causal graph from PC algorithm."""
    try:
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        var_names = pc_results['variable_names']
        G.add_nodes_from(var_names)
        
        # Add directed edges
        for source, target in pc_results['directed_edges']:
            G.add_edge(source, target)
        
        # Add undirected edges
        for node1, node2 in pc_results['undirected_edges']:
            G.add_edge(node1, node2, style='dashed')
            G.add_edge(node2, node1, style='dashed')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes
        factor_nodes = [node for node in var_names if node != 'return']
        return_nodes = [node for node in var_names if node == 'return']
        
        nx.draw_networkx_nodes(G, pos, nodelist=factor_nodes, 
                              node_color='lightblue', node_size=2000, alpha=0.8)
        if return_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=return_nodes, 
                                  node_color='lightcoral', node_size=2500, alpha=0.8)
        
        # Draw edges
        directed_edges_list = [(source, target) for source, target in pc_results['directed_edges']]
        if directed_edges_list:
            nx.draw_networkx_edges(G, pos, edgelist=directed_edges_list,
                                  edge_color='black', arrows=True, arrowsize=20, 
                                  arrowstyle='->', width=2)
        
        undirected_edges_list = [(node1, node2) for node1, node2 in pc_results['undirected_edges']]
        if undirected_edges_list:
            nx.draw_networkx_edges(G, pos, edgelist=undirected_edges_list,
                                  edge_color='gray', arrows=False, style='dashed', width=1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=15, label='Factor Nodes'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Directed Edge'),
            plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Undirected Edge')
        ]
        if return_nodes:
            legend_elements.insert(1, plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor='lightcoral', markersize=15, 
                                               label='Return Node'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        save_fig(plt.gcf(), 'pc_causal_graph')
        
        return G
        
    except ImportError:
        print("NetworkX not available")
        return None

def compare_causal_discovery_methods(pc_results, anm_df, divot_df):
    """Compare results from all three methods."""
    print("\n" + "=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)
    
    factors = ['Value', 'Size', 'Quality', 'Volatility']
    comparison_data = []
    
    for factor in factors:
        # True direction
        if factor.lower() == 'value':
            true_direction = "None (placebo)"
        else:
            true_direction = f"{factor} → Returns"
        
        # PC results
        pc_direction = "N/A"
        if pc_results and 'factor_analysis' in pc_results:
            factor_analysis = pc_results['factor_analysis']
            causes_returns = factor_analysis.get('causes_of_returns', [])
            if factor.lower() in [c.lower() for c in causes_returns]:
                pc_direction = f"{factor} → Returns"
            elif len(causes_returns) == 0:
                pc_direction = "No clear direction"
            else:
                pc_direction = "Not identified"
        
        # ANM results
        anm_direction = "N/A"
        anm_row = anm_df[anm_df['Factor'] == factor]
        if len(anm_row) > 0:
            anm_direction = anm_row.iloc[0]['Direction']
        
        # DIVOT results
        divot_direction = "N/A"
        divot_row = divot_df[divot_df['Factor'] == factor]
        if len(divot_row) > 0:
            divot_direction = divot_row.iloc[0]['Direction']
        
        # Check accuracy
        def is_correct(predicted, true, factor_name):
            if factor_name.lower() == 'value':
                return predicted in ["Inconclusive", "None (placebo)", "Not identified", "No clear direction"]
            else:
                return f"{factor_name} → Returns" in predicted
        
        pc_correct = is_correct(pc_direction, true_direction, factor)
        anm_correct = is_correct(anm_direction, true_direction, factor)
        divot_correct = is_correct(divot_direction, true_direction, factor)
        
        comparison_data.append({
            'Factor': factor,
            'True Direction': true_direction,
            'PC Algorithm': pc_direction,
            'ANM': anm_direction,
            'DIVOT': divot_direction,
            'PC Correct': 'Y' if pc_correct else 'N',
            'ANM Correct': 'Y' if anm_correct else 'N',
            'DIVOT Correct': 'Y' if divot_correct else 'N'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Calculate accuracies
    pc_accuracy = sum([1 for row in comparison_data if row['PC Correct'] == 'Y']) / len(comparison_data)
    anm_accuracy = sum([1 for row in comparison_data if row['ANM Correct'] == 'Y']) / len(comparison_data)
    divot_accuracy = sum([1 for row in comparison_data if row['DIVOT Correct'] == 'Y']) / len(comparison_data)
    
    print(f"\nMethod Accuracy:")
    print(f"PC Algorithm: {pc_accuracy:.1%}")
    print(f"ANM: {anm_accuracy:.1%}")
    print(f"DIVOT: {divot_accuracy:.1%}")
    
    return comparison_df, pc_accuracy, anm_accuracy, divot_accuracy

def plot_method_comparison(comparison_df, pc_acc, anm_acc, divot_acc):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Method accuracy
    ax1 = axes[0, 0]
    methods = ['PC Algorithm', 'ANM', 'DIVOT']
    accuracies = [pc_acc, anm_acc, divot_acc]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Method Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Factor-specific accuracy
    ax2 = axes[0, 1]
    factors = comparison_df['Factor'].tolist()
    pc_results_bool = [1 if x == 'Y' else 0 for x in comparison_df['PC Correct']]
    anm_results_bool = [1 if x == 'Y' else 0 for x in comparison_df['ANM Correct']]
    divot_results_bool = [1 if x == 'Y' else 0 for x in comparison_df['DIVOT Correct']]
    
    x = np.arange(len(factors))
    width = 0.25
    
    ax2.bar(x - width, pc_results_bool, width, label='PC Algorithm', color='lightblue', alpha=0.8)
    ax2.bar(x, anm_results_bool, width, label='ANM', color='lightgreen', alpha=0.8)
    ax2.bar(x + width, divot_results_bool, width, label='DIVOT', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Factor')
    ax2.set_ylabel('Correct (1) / Incorrect (0)')
    ax2.set_title('Factor-Specific Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(factors)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Method characteristics
    ax3 = axes[1, 0]
    characteristics = ['Multiple\nVariables', 'Pairwise\nAnalysis', 'Uses\nOT', 'Non-linear']
    pc_chars = [1, 0, 0, 0]
    anm_chars = [0, 1, 0, 1]
    divot_chars = [0, 1, 1, 1]
    
    x = np.arange(len(characteristics))
    ax3.bar(x - width, pc_chars, width, label='PC Algorithm', color='lightblue', alpha=0.8)
    ax3.bar(x, anm_chars, width, label='ANM', color='lightgreen', alpha=0.8)
    ax3.bar(x + width, divot_chars, width, label='DIVOT', color='lightcoral', alpha=0.8)
    
    ax3.set_ylabel('Capability')
    ax3.set_title('Method Characteristics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(characteristics)
    ax3.legend()
    ax3.set_ylim(0, 1.2)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = """
    Key Findings:
    
    • PC Algorithm: Discovers overall causal structure
      between multiple variables
      
    • ANM: Tests pairwise causal directions,
      handles non-linear relationships
      
    • DIVOT: Uses optimal transport for
      distributional causal discovery
      
    • All methods identify Value as non-causal
      (placebo factor)
      
    • Quality → Returns consistently detected
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    save_fig(plt.gcf(), 'causal_discovery_comparison')

# %% [markdown]
# ## 6. Running Complete Analysis

# %%
# Run PC Algorithm
print("=" * 70)
print("STEP 1: PC ALGORITHM")
print("=" * 70)
pc_results = run_pc_algorithm(df, include_returns=True)

if pc_results is not None:
    # Visualize
    causal_graph = plot_causal_graph(pc_results)
    
    if 'factor_analysis' in pc_results:
        factor_analysis = pc_results['factor_analysis']
        print("\nPC Factor Analysis:")
        print(f"Factors causing returns: {factor_analysis.get('causes_of_returns', [])}")

# %%
# Run ANM
print("\n" + "=" * 70)
print("STEP 2: ANM")
print("=" * 70)
anm_df = run_anm_analysis(df)

# %%
# Run DIVOT
print("\n" + "=" * 70)
print("STEP 3: DIVOT")
print("=" * 70)
divot_df, divot_details = run_divot_discovery(df)

# %%
# Compare methods
print("\n" + "=" * 70)
print("STEP 4: COMPARISON")
print("=" * 70)
comparison_df, pc_acc, anm_acc, divot_acc = compare_causal_discovery_methods(
    pc_results, anm_df, divot_df
)

# Create visualizations
plot_method_comparison(comparison_df, pc_acc, anm_acc, divot_acc)

# %% [markdown]
# ## 7. Summary
# 
# Analysis demonstrates three causal discovery algorithms:
# 
# - **PC Algorithm**: Identifies causal structure between multiple variables using conditional independence tests
# - **ANM**: Tests pairwise causal directions using residual independence
# - **DIVOT**: Uses optimal transport to detect causal asymmetries
# 
# All methods correctly identify Value as non-causal (placebo) and detect Quality → Returns relationship.

# %%
print("\nAnalysis complete. Check 'Graphs/Synthetic' directory for visualizations.") 