import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')

# Define colors
quality_color = '#2E86AB'
treatment_color = '#A23B72'
returns_color = '#F18F01'
confounder_color = '#C73E1D'

# Draw main nodes
# Quality factor
quality_box = FancyBboxPatch((0.5, 4), 2, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=quality_color, 
                            edgecolor='black', 
                            linewidth=2,
                            alpha=0.8)
ax.add_patch(quality_box)
ax.text(1.5, 4.6, 'Quality\nFactor', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# Treatment
treatment_box = FancyBboxPatch((4, 5.5), 2, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=treatment_color, 
                              edgecolor='black', 
                              linewidth=2,
                              alpha=0.8)
ax.add_patch(treatment_box)
ax.text(5, 6.1, 'Treatment\n(Policy)', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# Returns
returns_box = FancyBboxPatch((7.5, 4), 2, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=returns_color, 
                            edgecolor='black', 
                            linewidth=2,
                            alpha=0.8)
ax.add_patch(returns_box)
ax.text(8.5, 4.6, 'Stock\nReturns', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# Draw arrows
# Quality → Returns
ax.annotate('', xy=(7.3, 4.6), xytext=(2.7, 4.6),
            arrowprops=dict(arrowstyle='->', lw=3, color=quality_color))
ax.text(5, 4.8, 'Direct Causal Effect\n(+1% per σ)', ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=quality_color)

# Quality → Treatment
ax.annotate('', xy=(4.2, 5.8), xytext=(2.3, 5.0),
            arrowprops=dict(arrowstyle='->', lw=3, color=confounder_color))
ax.text(3, 5.7, 'Selection Bias\n(High quality → \nMore likely treated)', 
        ha='center', va='center', fontsize=10, fontweight='bold', 
        color=confounder_color, bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor='white', 
                                         edgecolor=confounder_color))

# Treatment → Returns
ax.annotate('', xy=(7.7, 5.5), xytext=(5.8, 6.0),
            arrowprops=dict(arrowstyle='->', lw=3, color=treatment_color))
ax.text(6.8, 6.2, 'Treatment Effect\n(+5% boost)', ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=treatment_color)

# Add confounding path indicator
confounder_path = patches.Arc((5, 3.5), 6, 4, angle=0, theta1=20, theta2=160,
                             color=confounder_color, linewidth=4, linestyle='--', alpha=0.7)
ax.add_patch(confounder_path)
ax.text(5, 2.5, 'CONFOUNDING PATH', ha='center', va='center', 
        fontsize=16, fontweight='bold', color=confounder_color,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                 edgecolor=confounder_color, linewidth=2))

# Add explanation text boxes
explanation_text = """CONFOUNDING PROBLEM:
• Naive comparison: Treated vs. Control returns
• Problem: Treated stocks have higher quality
• Quality also increases returns directly
• Simple comparison conflates treatment and quality effects"""

ax.text(0.5, 1.5, explanation_text, ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffe6e6', 
                 edgecolor=confounder_color, linewidth=1.5))

solution_text = """CAUSAL INFERENCE SOLUTIONS:
• Difference-in-Differences: Controls for baseline differences
• Matching: Pair similar quality stocks with/without treatment  
• Instrumental Variables: Use exogenous variation
• Optimal Transport: Minimize distributional imbalances"""

ax.text(5.5, 1.5, solution_text, ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='#e6f3ff', 
                 edgecolor=quality_color, linewidth=1.5))

# Add title
ax.text(5, 8.3, 'CONFOUNDING IN FACTOR INVESTING', ha='center', va='center',
        fontsize=18, fontweight='bold')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color=quality_color, lw=3, label='Direct Factor Effect'),
    plt.Line2D([0], [0], color=treatment_color, lw=3, label='Treatment Effect'),
    plt.Line2D([0], [0], color=confounder_color, lw=3, linestyle='--', label='Confounding Path'),
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
         fontsize=12, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('../Confounder.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Confounder graph saved as '../Confounder.png'") 