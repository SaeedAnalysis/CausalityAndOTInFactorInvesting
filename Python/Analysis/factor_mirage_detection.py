"""
Factor Mirage Detection Preprocessing - López de Prado Protocol
===============================================================

This module implements factor mirage detection following López de Prado's protocol
for identifying spurious factors in financial markets.

Key Features:
- Structural break detection
- Stability testing across time periods  
- Collider bias detection
- Confounder identification
- Statistical significance testing

Based on "A Protocol for Causal Factor Investing" by López de Prado
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.stattools import jarque_bera
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some tests will use simplified versions.")

class FactorMirageDetector:
    """
    Factor Mirage Detection following López de Prado's protocol
    
    This class implements comprehensive preprocessing to detect causally misspecified
    factor models before applying causal discovery algorithms.
    """
    
    def __init__(self, significance_level=0.05, lookback_window=24, min_observations=50):
        """
        Initialize Factor Mirage Detector
        
        Args:
            significance_level: Statistical significance threshold
            lookback_window: Rolling window for structural break detection
            min_observations: Minimum observations required for reliable tests
        """
        self.significance_level = significance_level
        self.lookback_window = lookback_window  
        self.min_observations = min_observations
        
    def detect_structural_breaks(self, factor_data, returns_data, dates=None):
        """
        Detect structural breaks in factor-return relationships
        
        López de Prado emphasizes that structural breaks can create spurious
        causal relationships when factors change behavior across regimes.
        
        Args:
            factor_data: Factor time series
            returns_data: Returns time series
            dates: Optional date index
            
        Returns:
            dict: Structural break analysis results
        """
        print("Detecting Structural Breaks in Factor-Return Relationships...")
        
        n_obs = len(factor_data)
        if n_obs < self.min_observations:
            return {'breaks_detected': False, 'reason': 'Insufficient data'}
        
        # Rolling correlation analysis
        window = self.lookback_window
        rolling_corr = []
        rolling_dates = []
        
        for i in range(window, n_obs):
            window_factor = factor_data[i-window:i]
            window_returns = returns_data[i-window:i]
            
            if np.std(window_factor) > 1e-6 and np.std(window_returns) > 1e-6:
                corr = np.corrcoef(window_factor, window_returns)[0, 1]
                rolling_corr.append(corr)
                if dates is not None:
                    rolling_dates.append(dates[i])
                else:
                    rolling_dates.append(i)
        
        rolling_corr = np.array(rolling_corr)
        
        # Detect significant changes in correlation
        # Use simple change point detection based on rolling standard deviation
        corr_changes = np.abs(np.diff(rolling_corr))
        threshold = 2 * np.std(corr_changes)
        
        break_points = np.where(corr_changes > threshold)[0]
        
        # Additional test: Chow test approximation
        # Compare first half vs second half correlations
        mid_point = len(rolling_corr) // 2
        first_half_corr = np.mean(rolling_corr[:mid_point])
        second_half_corr = np.mean(rolling_corr[mid_point:])
        
        # Statistical test for correlation difference
        corr_diff = abs(second_half_corr - first_half_corr)
        corr_diff_threshold = 0.2  # López de Prado suggests 0.2 as meaningful threshold
        
        structural_break_detected = len(break_points) > 0 or corr_diff > corr_diff_threshold
        
        results = {
            'breaks_detected': structural_break_detected,
            'n_breakpoints': len(break_points),
            'breakpoint_indices': break_points,
            'rolling_correlation': rolling_corr,
            'rolling_dates': rolling_dates,
            'first_half_correlation': first_half_corr,
            'second_half_correlation': second_half_corr,
            'correlation_shift': corr_diff,
            'shift_significant': corr_diff > corr_diff_threshold,
            'recommendation': 'CAUTION: Structural breaks detected' if structural_break_detected else 'OK: No major structural breaks'
        }
        
        print(f"  Structural Break Analysis:")
        print(f"    Break points detected: {len(break_points)}")
        print(f"    Correlation shift: {corr_diff:.3f}")
        print(f"    Recommendation: {results['recommendation']}")
        
        return results
    
    def detect_serial_dependence(self, factor_data, returns_data):
        """
        Detect autocorrelation and serial dependence that can confound causal inference
        
        López de Prado emphasizes that serial correlation can create spurious leads/lags
        that are mistaken for causal relationships.
        
        Args:
            factor_data: Factor time series
            returns_data: Returns time series
            
        Returns:
            dict: Serial dependence analysis results
        """
        print("Detecting Serial Dependence and Autocorrelation...")
        
        results = {}
        
        # Factor autocorrelation
        factor_autocorr = []
        returns_autocorr = []
        
        # Calculate autocorrelations up to lag 5
        for lag in range(1, 6):
            if len(factor_data) > lag:
                factor_lag_corr = np.corrcoef(factor_data[:-lag], factor_data[lag:])[0, 1]
                returns_lag_corr = np.corrcoef(returns_data[:-lag], returns_data[lag:])[0, 1]
                
                factor_autocorr.append(factor_lag_corr)
                returns_autocorr.append(returns_lag_corr)
        
        # Ljung-Box test for serial correlation (if available)
        if STATSMODELS_AVAILABLE and len(factor_data) >= 20:
            try:
                factor_ljung = acorr_ljungbox(factor_data, lags=min(10, len(factor_data)//4), return_df=True)
                returns_ljung = acorr_ljungbox(returns_data, lags=min(10, len(returns_data)//4), return_df=True)
                
                factor_serial_corr = factor_ljung['lb_pvalue'].min() < self.significance_level
                returns_serial_corr = returns_ljung['lb_pvalue'].min() < self.significance_level
                
            except:
                factor_serial_corr = max([abs(x) for x in factor_autocorr]) > 0.3
                returns_serial_corr = max([abs(x) for x in returns_autocorr]) > 0.3
        else:
            # Simple threshold test
            factor_serial_corr = max([abs(x) for x in factor_autocorr]) > 0.3
            returns_serial_corr = max([abs(x) for x in returns_autocorr]) > 0.3
        
        # Cross-lagged correlations (lead-lag relationships)
        cross_lags = {}
        for lag in range(1, 4):
            if len(factor_data) > lag:
                # Factor leads returns
                factor_leads = np.corrcoef(factor_data[:-lag], returns_data[lag:])[0, 1]
                # Returns lead factor
                returns_leads = np.corrcoef(returns_data[:-lag], factor_data[lag:])[0, 1]
                
                cross_lags[f'factor_leads_{lag}'] = factor_leads
                cross_lags[f'returns_leads_{lag}'] = returns_leads
        
        # Mirage risk from serial dependence
        serial_mirage_risk = 'High' if (factor_serial_corr or returns_serial_corr) else 'Low'
        
        results = {
            'factor_autocorrelations': factor_autocorr,
            'returns_autocorrelations': returns_autocorr,
            'factor_serial_correlation': factor_serial_corr,
            'returns_serial_correlation': returns_serial_corr,
            'cross_lagged_correlations': cross_lags,
            'serial_mirage_risk': serial_mirage_risk,
            'recommendation': 'CAUTION: Serial dependence detected' if serial_mirage_risk == 'High' else 'OK: Limited serial dependence'
        }
        
        print(f"  Serial Dependence Analysis:")
        print(f"    Factor autocorrelation (max): {max([abs(x) for x in factor_autocorr]):.3f}")
        print(f"    Returns autocorrelation (max): {max([abs(x) for x in returns_autocorr]):.3f}")
        print(f"    Serial mirage risk: {serial_mirage_risk}")
        print(f"    Recommendation: {results['recommendation']}")
        
        return results
    
    def detect_collider_bias(self, factor_data, returns_data, other_factors=None):
        """
        Advanced collider bias detection with interaction effects
        
        López de Prado emphasizes that colliders can create spurious correlations
        when included as controls. This extends basic collider detection with
        interaction effects and non-linear relationships.
        
        Args:
            factor_data: Primary factor
            returns_data: Returns
            other_factors: Dict of other factors to test as potential colliders
            
        Returns:
            dict: Advanced collider bias analysis
        """
        print("Detecting Advanced Collider Bias with Interactions...")
        
        if other_factors is None:
            return {'collider_bias_detected': False, 'reason': 'No other factors provided'}
        
        collider_results = {}
        
        # Baseline correlation
        baseline_corr = np.corrcoef(factor_data, returns_data)[0, 1]
        
        for collider_name, collider_data in other_factors.items():
            print(f"  Testing {collider_name} as potential collider...")
            
            # Standard collider test
            try:
                # Partial correlation controlling for potential collider
                # Using linear regression residuals
                
                # Regression of factor on collider
                X_collider = collider_data.reshape(-1, 1)
                model_factor = LinearRegression().fit(X_collider, factor_data)
                factor_residuals = factor_data - model_factor.predict(X_collider)
                
                # Regression of returns on collider
                model_returns = LinearRegression().fit(X_collider, returns_data)
                returns_residuals = returns_data - model_returns.predict(X_collider)
                
                # Partial correlation
                partial_corr = np.corrcoef(factor_residuals, returns_residuals)[0, 1]
                
                # Collider detection criteria
                corr_increase = abs(partial_corr) > abs(baseline_corr)  # Correlation increases
                sign_flip = np.sign(partial_corr) != np.sign(baseline_corr)  # Sign flips
                large_change = abs(partial_corr - baseline_corr) > 0.1  # Large change
                
                # Interaction effect test
                # Test if factor*collider interaction is significant
                factor_std = (factor_data - np.mean(factor_data)) / np.std(factor_data)
                collider_std = (collider_data - np.mean(collider_data)) / np.std(collider_data)
                interaction_term = factor_std * collider_std
                
                # Regression with interaction
                X_interaction = np.column_stack([factor_std, collider_std, interaction_term])
                model_interaction = LinearRegression().fit(X_interaction, returns_data)
                interaction_coef = model_interaction.coef_[2]  # Interaction coefficient
                
                # Statistical significance of interaction (approximate)
                predictions = model_interaction.predict(X_interaction)
                residuals = returns_data - predictions
                se = np.sqrt(np.sum(residuals**2) / (len(returns_data) - 4))
                interaction_t_stat = abs(interaction_coef) / se
                interaction_significant = interaction_t_stat > 2  # Approximate t-test
                
                # Collider score
                collider_score = 0
                if corr_increase: collider_score += 2
                if sign_flip: collider_score += 3
                if large_change: collider_score += 1
                if interaction_significant: collider_score += 2
                
                collider_results[collider_name] = {
                    'baseline_correlation': baseline_corr,
                    'partial_correlation': partial_corr,
                    'correlation_increase': corr_increase,
                    'sign_flip': sign_flip,
                    'large_change': large_change,
                    'interaction_coefficient': interaction_coef,
                    'interaction_significant': interaction_significant,
                    'collider_score': collider_score,
                    'collider_risk': 'High' if collider_score >= 4 else 'Medium' if collider_score >= 2 else 'Low'
                }
                
                print(f"    Baseline corr: {baseline_corr:.3f} → Partial corr: {partial_corr:.3f}")
                print(f"    Interaction coef: {interaction_coef:.3f} (significant: {interaction_significant})")
                print(f"    Collider risk: {collider_results[collider_name]['collider_risk']}")
                
            except Exception as e:
                print(f"    Error testing {collider_name}: {e}")
                collider_results[collider_name] = {
                    'collider_risk': 'Unknown',
                    'error': str(e)
                }
        
        # Overall assessment
        high_risk_colliders = [name for name, result in collider_results.items() 
                              if result.get('collider_risk') == 'High']
        
        overall_result = {
            'collider_results': collider_results,
            'high_risk_colliders': high_risk_colliders,
            'collider_bias_detected': len(high_risk_colliders) > 0,
            'recommendation': f'AVOID as controls: {high_risk_colliders}' if high_risk_colliders else 'OK: No high-risk colliders detected'
        }
        
        return overall_result
    
    def detect_confounder_bias(self, factor_data, returns_data, other_factors=None, external_variables=None):
        """
        Advanced confounder detection with proxy variables and principal components
        
        López de Prado emphasizes detecting hidden confounders that cause both
        factors and returns. This uses proxy variables and PCA to detect potential
        latent confounders.
        
        Args:
            factor_data: Primary factor
            returns_data: Returns
            other_factors: Dict of other factors (potential confounders)
            external_variables: Dict of external variables (economic indicators, etc.)
            
        Returns:
            dict: Advanced confounder analysis
        """
        print("Detecting Advanced Confounder Bias with Proxies...")
        
        confounder_results = {}
        
        # Combine other factors and external variables
        all_potential_confounders = {}
        if other_factors:
            all_potential_confounders.update(other_factors)
        if external_variables:
            all_potential_confounders.update(external_variables)
        
        if not all_potential_confounders:
            return {'confounder_bias_detected': False, 'reason': 'No potential confounders provided'}
        
        for confounder_name, confounder_data in all_potential_confounders.items():
            print(f"  Testing {confounder_name} as potential confounder...")
            
            try:
                # Confounder criteria:
                # 1. Correlated with factor
                # 2. Correlated with returns  
                # 3. When controlled for, factor-return relationship changes significantly
                
                factor_confounder_corr = np.corrcoef(factor_data, confounder_data)[0, 1]
                returns_confounder_corr = np.corrcoef(returns_data, confounder_data)[0, 1]
                
                # Multiple regression controlling for confounder
                X_controlled = np.column_stack([factor_data, confounder_data])
                model_controlled = LinearRegression().fit(X_controlled, returns_data)
                controlled_coef = model_controlled.coef_[0]  # Factor coefficient
                
                # Simple regression without confounder
                model_simple = LinearRegression().fit(factor_data.reshape(-1, 1), returns_data)
                simple_coef = model_simple.coef_[0]
                
                # Confounder detection
                coef_change = abs(controlled_coef - simple_coef)
                large_coef_change = coef_change > 0.1 * abs(simple_coef)
                sign_change = np.sign(controlled_coef) != np.sign(simple_coef)
                
                # Confounder score
                confounder_score = 0
                if abs(factor_confounder_corr) > 0.2: confounder_score += 1
                if abs(returns_confounder_corr) > 0.2: confounder_score += 1  
                if large_coef_change: confounder_score += 2
                if sign_change: confounder_score += 3
                if abs(factor_confounder_corr) > 0.3 and abs(returns_confounder_corr) > 0.3: confounder_score += 2
                
                confounder_results[confounder_name] = {
                    'factor_correlation': factor_confounder_corr,
                    'returns_correlation': returns_confounder_corr,
                    'simple_coefficient': simple_coef,
                    'controlled_coefficient': controlled_coef,
                    'coefficient_change': coef_change,
                    'large_change': large_coef_change,
                    'sign_change': sign_change,
                    'confounder_score': confounder_score,
                    'confounder_risk': 'High' if confounder_score >= 5 else 'Medium' if confounder_score >= 3 else 'Low'
                }
                
                print(f"    Factor corr: {factor_confounder_corr:.3f}, Returns corr: {returns_confounder_corr:.3f}")
                print(f"    Coef change: {simple_coef:.3f} → {controlled_coef:.3f}")
                print(f"    Confounder risk: {confounder_results[confounder_name]['confounder_risk']}")
                
            except Exception as e:
                print(f"    Error testing {confounder_name}: {e}")
                confounder_results[confounder_name] = {
                    'confounder_risk': 'Unknown',
                    'error': str(e)
                }
        
        # PCA-based latent confounder detection
        latent_confounder_analysis = self._detect_latent_confounders(factor_data, returns_data, all_potential_confounders)
        
        # Overall assessment
        high_risk_confounders = [name for name, result in confounder_results.items() 
                                if result.get('confounder_risk') == 'High']
        
        overall_result = {
            'confounder_results': confounder_results,
            'latent_confounder_analysis': latent_confounder_analysis,
            'high_risk_confounders': high_risk_confounders,
            'confounder_bias_detected': len(high_risk_confounders) > 0 or latent_confounder_analysis['latent_confounders_detected'],
            'recommendation': f'INCLUDE as controls: {high_risk_confounders}' if high_risk_confounders else 'OK: No major confounders detected'
        }
        
        return overall_result
    
    def _detect_latent_confounders(self, factor_data, returns_data, other_variables):
        """
        Use PCA to detect potential latent confounders
        """
        try:
            # Combine all variables for PCA
            all_data = np.column_stack([factor_data, returns_data] + list(other_variables.values()))
            
            # Standardize
            scaler = StandardScaler()
            all_data_std = scaler.fit_transform(all_data)
            
            # PCA
            pca = PCA()
            pca.fit(all_data_std)
            
            # Look for principal components that explain significant variance
            # and are correlated with both factor and returns
            explained_variance = pca.explained_variance_ratio_
            components = pca.components_
            
            latent_confounders_detected = False
            significant_components = []
            
            for i, (var_ratio, component) in enumerate(zip(explained_variance, components)):
                if var_ratio > 0.1:  # Component explains >10% of variance
                    factor_loading = abs(component[0])  # Factor loading
                    returns_loading = abs(component[1])  # Returns loading
                    
                    # Check if component loads significantly on both factor and returns
                    if factor_loading > 0.3 and returns_loading > 0.3:
                        latent_confounders_detected = True
                        significant_components.append({
                            'component': i,
                            'variance_explained': var_ratio,
                            'factor_loading': factor_loading,
                            'returns_loading': returns_loading
                        })
            
            return {
                'latent_confounders_detected': latent_confounders_detected,
                'significant_components': significant_components,
                'total_explained_variance': sum(explained_variance[:3])  # First 3 components
            }
            
        except Exception as e:
            return {
                'latent_confounders_detected': False,
                'error': str(e)
            }
    
    def compute_comprehensive_mirage_score(self, factor_name, factor_data, returns_data, other_factors=None, dates=None):
        """
        Compute comprehensive factor mirage score following López de Prado protocol
        
        Args:
            factor_name: Name of the factor
            factor_data: Factor time series
            returns_data: Returns time series
            other_factors: Dict of other factors for collider/confounder testing
            dates: Optional date index
            
        Returns:
            dict: Comprehensive mirage analysis with preprocessing recommendations
        """
        print(f"\nMIRAGE ANALYSIS SUMMARY:")
        print("=" * 70)
        print("Following López de Prado's Causal Factor Investing Protocol")
        
        # Run all mirage detection tests
        structural_breaks = self.detect_structural_breaks(factor_data, returns_data, dates)
        serial_dependence = self.detect_serial_dependence(factor_data, returns_data)
        
        collider_bias = {'collider_bias_detected': False}
        confounder_bias = {'confounder_bias_detected': False}
        
        if other_factors:
            collider_bias = self.detect_collider_bias(factor_data, returns_data, other_factors)
            confounder_bias = self.detect_confounder_bias(factor_data, returns_data, other_factors)
        
        # Compute overall mirage risk score
        mirage_score = 0
        
        # Structural breaks (0-3 points)
        if structural_breaks['breaks_detected']:
            if structural_breaks['shift_significant']:
                mirage_score += 3
            else:
                mirage_score += 1
        
        # Serial dependence (0-2 points)
        if serial_dependence['serial_mirage_risk'] == 'High':
            mirage_score += 2
        
        # Collider bias (0-3 points)
        if collider_bias['collider_bias_detected']:
            high_risk_colliders = len(collider_bias.get('high_risk_colliders', []))
            mirage_score += min(high_risk_colliders, 3)
        
        # Confounder bias (0-2 points)
        if confounder_bias['confounder_bias_detected']:
            mirage_score += 2
        
        # Overall risk level
        if mirage_score >= 7:
            overall_risk = 'VERY HIGH'
        elif mirage_score >= 5:
            overall_risk = 'HIGH'
        elif mirage_score >= 3:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        # López de Prado preprocessing recommendations
        recommendations = []
        
        if structural_breaks['breaks_detected']:
            recommendations.append("Apply regime-aware modeling or exclude break periods")
        
        if serial_dependence['serial_mirage_risk'] == 'High':
            recommendations.append("Use Newey-West robust standard errors or first-difference data")
        
        if collider_bias['collider_bias_detected']:
            avoid_controls = collider_bias.get('high_risk_colliders', [])
            if avoid_controls:
                recommendations.append(f"AVOID these variables as controls: {avoid_controls}")
        
        if confounder_bias['confounder_bias_detected']:
            include_controls = confounder_bias.get('high_risk_confounders', [])
            if include_controls:
                recommendations.append(f"INCLUDE these variables as controls: {include_controls}")
        
        if not recommendations:
            recommendations.append("Proceed with standard causal analysis - low mirage risk detected")
        
        # Final results
        results = {
            'factor_name': factor_name,
            'mirage_score': mirage_score,
            'overall_risk': overall_risk,
            'structural_breaks': structural_breaks,
            'serial_dependence': serial_dependence,
            'collider_bias': collider_bias,
            'confounder_bias': confounder_bias,
            'recommendations': recommendations,
            'proceed_with_causal_analysis': overall_risk in ['LOW', 'MEDIUM'],
            'preprocessing_required': overall_risk in ['HIGH', 'VERY HIGH']
        }
        
        # Summary output
        print(f"\n📊 MIRAGE ANALYSIS SUMMARY:")
        print(f"   Factor: {factor_name}")
        print(f"   Mirage Score: {mirage_score}/10")
        print(f"   Overall Risk: {overall_risk}")
        print(f"   Proceed with Causal Analysis: {'YES' if results['proceed_with_causal_analysis'] else 'NO - PREPROCESSING REQUIRED'}")
        
        print(f"\nLÓPEZ DE PRADO RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return results


def enhanced_mirage_preprocessing_pipeline(data_dict, returns_column='return', factor_columns=None):
    """
    Run enhanced factor mirage detection on multiple factors
    
    Args:
        data_dict: Dictionary with factor data and returns
        returns_column: Name of returns column
        factor_columns: List of factor columns to analyze (None = all except returns)
        
    Returns:
        dict: Mirage analysis results for all factors with preprocessing recommendations
    """
    print("\nFACTOR MIRAGE PREPROCESSING PIPELINE")
    print("=" * 80)
    print("Implementing López de Prado's Causal Factor Investing Protocol")
    
    if returns_column not in data_dict:
        raise ValueError(f"Returns column '{returns_column}' not found in data")
    
    returns_data = data_dict[returns_column]
    
    if factor_columns is None:
        factor_columns = [col for col in data_dict.keys() if col != returns_column and col != 'dates']
    
    # Initialize detector
    detector = EnhancedFactorMirageDetector()
    
    # Run mirage detection for each factor
    all_results = {}
    
    for factor in factor_columns:
        if factor not in data_dict:
            print(f"Warning: Factor '{factor}' not found in data, skipping...")
            continue
        
        factor_data = data_dict[factor]
        other_factors = {k: v for k, v in data_dict.items() 
                        if k != factor and k != returns_column and k != 'dates'}
        
        dates = data_dict.get('dates', None)
        
        # Run comprehensive mirage analysis
        results = detector.compute_comprehensive_mirage_score(
            factor, factor_data, returns_data, other_factors, dates
        )
        
        all_results[factor] = results
    
    # Summary across all factors
    print(f"\nPIPELINE SUMMARY - ALL FACTORS")
    print("=" * 50)
    
    high_risk_factors = [f for f, r in all_results.items() if r['overall_risk'] in ['HIGH', 'VERY HIGH']]
    medium_risk_factors = [f for f, r in all_results.items() if r['overall_risk'] == 'MEDIUM']
    low_risk_factors = [f for f, r in all_results.items() if r['overall_risk'] == 'LOW']
    
    print(f"High Risk Factors (preprocessing required): {high_risk_factors}")
    print(f"Medium Risk Factors (proceed with caution): {medium_risk_factors}")
    print(f"Low Risk Factors (proceed with standard analysis): {low_risk_factors}")
    
    # Global recommendations
    print(f"\nGLOBAL PREPROCESSING RECOMMENDATIONS:")
    
    if high_risk_factors:
        print(f"1. Address mirage issues in {high_risk_factors} before causal analysis")
        print(f"2. Consider regime-aware modeling for structural breaks")
        print(f"3. Apply robust standard errors for serial correlation")
    
    if medium_risk_factors:
        print(f"4. Monitor {medium_risk_factors} for sensitivity to specification changes")
    
    if low_risk_factors:
        print(f"5. {low_risk_factors} cleared for standard causal discovery methods")
    
    pipeline_result = {
        'individual_results': all_results,
        'high_risk_factors': high_risk_factors,
        'medium_risk_factors': medium_risk_factors,
        'low_risk_factors': low_risk_factors,
        'global_recommendations': {
            'high_risk_count': len(high_risk_factors),
            'preprocessing_required': len(high_risk_factors) > 0,
            'proceed_with_caution': len(medium_risk_factors) > 0,
            'ready_for_analysis': len(low_risk_factors)
        }
    }
    
    return pipeline_result


if __name__ == "__main__":
    # Demonstration with synthetic data
    print("Factor Mirage Detection Demonstration")
    
    # Generate synthetic data with known mirage issues
    np.random.seed(42)
    n_obs = 200
    
    # Create structural break in the middle
    regime1 = np.random.normal(0, 1, n_obs//2)
    regime2 = np.random.normal(2, 1, n_obs//2)  # Structural break
    factor_with_break = np.concatenate([regime1, regime2])
    
    # Create serial correlation
    factor_serial = np.random.normal(0, 1, n_obs)
    for i in range(1, n_obs):
        factor_serial[i] += 0.5 * factor_serial[i-1]  # AR(1)
    
    # Create collider
    factor_clean = np.random.normal(0, 1, n_obs)
    returns = 0.3 * factor_clean + np.random.normal(0, 0.5, n_obs)
    collider = 0.4 * factor_clean + 0.3 * returns + np.random.normal(0, 0.3, n_obs)
    
    demo_data = {
        'factor_with_break': factor_with_break,
        'factor_with_serial': factor_serial,
        'clean_factor': factor_clean,
        'collider_factor': collider,
        'return': returns
    }
    
    # Run pipeline
    results = enhanced_mirage_preprocessing_pipeline(demo_data)
    
    print("\nFactor Mirage Detection demonstration completed!") 