# Graph Interpretations for Real Data Analysis

## 1. factor_distributions_real_data.png

**What It Shows**: Distribution of monthly returns for SMB, HML, Momentum, and RMW factors

**How to Read**: X-axis shows return values (decimal), Y-axis shows frequency. Histogram bars show actual data, smooth curve shows kernel density estimate.

**Key Insights**: Look for: (1) Symmetry vs skewness, (2) Fat tails indicating extreme events, (3) Center of distribution relative to zero

**Interpretation of Results**: Most factors show roughly normal distributions with some fat tails, indicating occasional extreme returns. This is typical for financial factors.

---

## 2. returns_time_real_data.png

**What It Shows**: Time series of average monthly excess returns for small cap vs large cap portfolios

**How to Read**: Blue line = small cap average returns, Red dashed line = large cap average returns. Y-axis in percentage.

**Key Insights**: Look for: (1) Periods of outperformance, (2) Volatility clustering, (3) Major market events impact

**Interpretation of Results**: Small caps show higher volatility but similar long-term returns. Notable divergence during crisis periods where large caps typically outperform.

---

## 3. correlation_matrix_real_data.png

**What It Shows**: Pairwise correlations between all factors and returns

**How to Read**: Blue = positive correlation, Red = negative correlation. Darker colors = stronger relationships. Values range from -1 to +1.

**Key Insights**: Look for: (1) Factor independence, (2) Which factors correlate with returns, (3) Potential multicollinearity issues

**Interpretation of Results**: Market factor shows strongest correlation with returns. HML and CMA are highly correlated (0.68), suggesting they capture similar effects. Most factors show low correlation with each other.

---

## 4. did_results_real_data.png

**What It Shows**: Difference-in-Differences analysis for the Dot-com bubble event

**How to Read**: Top left: time series of treated (value) vs control (growth) returns. Top right: pre/post averages. Bottom: distribution changes.

**Key Insights**: DiD estimate shows causal effect. Positive = treated group benefited. Check parallel trends assumption in pre-period.

**Interpretation of Results**: DiD estimate of 0.99% shows value stocks outperformed growth stocks by about 1% due to the dot-com crash, confirming the value premium during market corrections.

---

## 5. covariate_balance_plot_real_data.png

**What It Shows**: Standardized mean differences between high and low momentum portfolios across other factors

**How to Read**: Green bars = good balance (<0.1), Red bars = imbalance (>0.1). Y-axis shows standardized difference (effect size).

**Key Insights**: Good balance ensures fair comparison. Large imbalances suggest selection bias that could confound results.

**Interpretation of Results**: Most factors show good balance between high/low momentum groups, validating comparisons. Any imbalances are within acceptable ranges.

---

## 6. iv_results_real_data.png

**What It Shows**: Comparison of OLS vs IV estimates for causal effects of factors on returns

**How to Read**: Blue bars = OLS estimates, Red bars = IV estimates. F-stat >10 indicates strong instrument. Large differences suggest endogeneity.

**Key Insights**: IV corrects for endogeneity bias. Weak instruments (low F) make IV unreliable. Compare magnitude and sign changes.

**Interpretation of Results**: SIZE shows dramatic difference (OLS: 0.89, IV: -8.34) but weak instrument (F=0.55) makes IV unreliable. VALUE and MOMENTUM have strong instruments and show meaningful endogeneity correction.

---

## 7. causal_graph_real_data.png

**What It Shows**: Network diagram of causal relationships between factors and returns based on theory and empirical analysis

**How to Read**: Arrows show causal direction. Solid = direct causation, Dashed = bidirectional/uncertain. All factors point to Returns node.

**Key Insights**: Direct paths show primary causal effects. Indirect paths through other factors show mediation effects.

**Interpretation of Results**: All factors show direct causal paths to returns (based on theory). HML-Momentum bidirectional relationship reflects known negative correlation between value and momentum strategies.

---

## 8. regime_effects_real_data.png

**What It Shows**: How factor effectiveness (regression coefficients) varies between high and low volatility market regimes

**How to Read**: Dark red = high volatility regime effects, Dark blue = low volatility regime effects. Y-axis shows coefficient magnitude.

**Key Insights**: Different heights show regime dependence. Sign changes indicate fundamental shifts in factor behavior.

**Interpretation of Results**: SIZE effect stronger in high vol (0.92 vs 0.80). MOMENTUM switches from negative in high vol to positive in low vol, suggesting it works better in calm markets. VALUE also shows regime dependence.

---

## 9. causal_discovery_comparison_real_data.png

**What It Shows**: Agreement between ANM and DIVOT causal discovery methods for each factor

**How to Read**: 1 = method found causal relationship, 0 = no causality found. Blue = ANM results, Green = DIVOT results.

**Key Insights**: Agreement between methods increases confidence. Both showing 0 suggests no clear causality or complex relationships.

**Interpretation of Results**: Both methods show inconclusive (0) results for all factors, indicating complex, possibly nonlinear or time-varying causal relationships that simple methods cannot detect in financial data.

---

## 10. factor_performance_summary_real_data.png

**What It Shows**: Cumulative performance of each factor over the full sample period (1963-2025)

**How to Read**: Each subplot shows one factor. Y-axis: cumulative return (1 = starting value). Text box shows annualized metrics.

**Key Insights**: Upward slope = positive returns. Volatility shown by line jaggedness. Sharpe ratio = risk-adjusted performance.

**Interpretation of Results**: Market factor shows strongest performance. SMB and HML show positive but volatile returns. Momentum shows high volatility. All factors exhibit significant drawdowns during crisis periods.

---

