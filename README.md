# Causal Discovery in Factor Investing with Optimal Transport

*An academic thesis project by Saeed Ali Nasser Alameri - For academic and research purposes only*

## Overview

This repository contains the code, data, and thesis materials for my Master's research project on applying causal discovery algorithms to factor investing, with a special focus on enhancements using optimal transport theory.

**Note: This work is shared for academic review and research purposes. Commercial use is prohibited without permission.**

## Project Structure

```
.
├── Graphs/                    # All generated visualizations
│   ├── Synthetic/            # Graphs from synthetic data analysis
│   └── Real Data/            # Graphs from Fama-French data analysis
│
├── Overleaf/                 # LaTeX source files for thesis
│   ├── Chapter-1.txt         # Introduction
│   ├── Chapter-2.txt         # Literature Review
│   ├── Chapter-3.txt         # Methodology and Results
│   ├── Conclusion.txt        # Conclusion and Future Work
│   ├── Bibliography.txt      # References in BibTeX format
│   └── Appendix.txt          # Appendices with detailed results
│
├── Python/                   # All code files
│   ├── Analysis/            # Core analysis scripts
│   │   ├── Causality_Main.py              # Synthetic data analysis
│   │   └── Causality_Real_Data.py         # Real data analysis (Fama-French)
│   ├── Visualization/       # Graph generation scripts
│   │   ├── Causality_Main_Graphs.py       # Synthetic data visualization
│   │   └── Causality_Real_Data_Graphs.py  # Real data visualization
│   ├── Jupyter/             # Jupyter notebook versions (if available)
│   └── run_all_analyses.py  # Main runner script
│
├── Real_Data/               # Fama-French data files
│   ├── F-F_Research_Data_Factors.csv
│   ├── F-F_Research_Data_5_Factors_2x3.csv
│   ├── F-F_Momentum_Factor.csv
│   └── 25_Portfolios_5x5.csv
│
└── Thesis Sources & References/  # Research papers and references
```

## Key Findings

### Synthetic Data Analysis
- Successfully validated causal inference methods on controlled data with known ground truth
- DiD and Changes-in-Changes provided most accurate treatment effect estimates (1.76% and 1.86% vs true 2%)
- ANM and DIVOT causal discovery methods each achieved 25% accuracy in identifying causal directions
- Optimal Transport matching showed improvements over traditional propensity score matching

### Real Data Analysis (Fama-French Factors)
- Applied same methods to real financial data from Kenneth French's data library (1963-2025)
- Natural experiments (Dot-com bubble, Financial Crisis) validated causal effects through DiD
- Factor effects showed strong regime dependence, with momentum reversing sign between high/low volatility periods
- Causal discovery methods returned uniformly inconclusive results, highlighting complexity of real financial markets

## Methods Implemented

### Causal Inference Techniques
1. **Difference-in-Differences (DiD)** - Both classical and OT-based distributional versions
2. **Matching Methods** - Propensity Score Matching and Optimal Transport Matching
3. **Instrumental Variables (IV)** - Two-stage least squares with synthetic instruments
4. **Changes-in-Changes (CiC)** - Distribution-aware treatment effect estimation

### Causal Discovery Algorithms
1. **Additive Noise Model (ANM)** - Tests independence of residuals to infer causality
2. **DIVOT** - Uses optimal transport to identify causal direction through volatility dynamics
3. **PC Algorithm** - Constraint-based causal graph discovery

## Requirements

```python
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
POT (Python Optimal Transport)  # Optional but recommended
causal-learn                    # Optional for PC algorithm
```

## Quick Start

### Run all analyses and generate graphs:
```bash
python Python/run_all_analyses.py
```

### Run specific analyses:
```bash
# Only synthetic data
python Python/run_all_analyses.py --synthetic-only

# Only real data  
python Python/run_all_analyses.py --real-only

# Only generate graphs (skip analysis)
python Python/run_all_analyses.py --graphs-only
```

### Run individual components:
```bash
# Synthetic data analysis
python Python/Analysis/Causality_Main.py

# Real data analysis
python Python/Analysis/Causality_Real_Data.py

# Generate synthetic graphs
python Python/Visualization/Causality_Main_Graphs.py

# Generate real data graphs
python Python/Visualization/Causality_Real_Data_Graphs.py
```

## Data Sources

- **Synthetic Data**: Generated using multivariate normal distributions with known causal structure
- **Real Data**: Fama-French factors and portfolio returns from [Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## Citation

If you use this code or methodology in your research, please cite:

```
Saeed Ali Alameri. (2025). Causal Discovery in Factor Investing with Optimal Transport. 
Master's Thesis, Khalifa University.
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

**Academic Use Only**: This thesis and associated code are provided for academic and research purposes. Commercial use is prohibited without explicit permission from the author.

- ✅ **Allowed**: Academic research, educational use, citation in papers
- ❌ **Not Allowed**: Commercial applications, proprietary use without permission

For full license details, see the LICENSE file or visit [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgments

- Kenneth R. French for providing the factor data
- Thesis advisor and committee members
- Authors of the causal-learn and POT libraries 