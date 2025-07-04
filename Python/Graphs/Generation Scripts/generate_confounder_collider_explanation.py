#!/usr/bin/env python3
"""
Generate Confounder and Collider Bias Explanation Graph
Based on López de Prado's "A Protocol for Causal Factor Investing"

This script creates a visualization explaining:
1. Confounder bias - when a variable influences both factor and returns
2. Collider bias - when controlling for a variable that's caused by both factor and returns
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_confounder_collider_explanation():
    """
    Create comprehensive explanation of confounder and collider bias in factor investing
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Confounder and Collider Bias in Factor Investing\nBased on López de Prado\'s Causal Factor Investing Protocol', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Colors
    factor_color = '#2E86AB'    # Blue
    return_color = '#A23B72'    # Purple
    confounder_color = '#F18F01'  # Orange
    collider_color = '#C73E1D'    # Red
    correct_color = '#4CAF50'     # Green
    wrong_color = '#FF5722'       # Red-Orange
    
    # ============================================================================
    # TOP ROW: CONFOUNDER BIAS EXPLANATION
    # ============================================================================
    
    # Subplot 1: True Causal Structure with Confounder
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('True Causal Structure\n(Confounder Present)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw nodes
    factor_circle = Circle((2, 5), 1.2, color=factor_color, alpha=0.8)
    return_circle = Circle((8, 5), 1.2, color=return_color, alpha=0.8)
    confounder_circle = Circle((5, 8), 1.2, color=confounder_color, alpha=0.8)
    
    ax1.add_patch(factor_circle)
    ax1.add_patch(return_circle)
    ax1.add_patch(confounder_circle)
    
    # Add labels
    ax1.text(2, 5, 'Book-to-\nMarket', ha='center', va='center', fontweight='bold', fontsize=10)
    ax1.text(8, 5, 'Returns', ha='center', va='center', fontweight='bold', fontsize=10)
    ax1.text(5, 8, 'Leverage\n(Confounder)', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw arrows
    # Confounder to Factor
    arrow1 = FancyArrowPatch((4.2, 7.2), (2.8, 5.8),
                            arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    ax1.add_patch(arrow1)
    
    # Confounder to Returns
    arrow2 = FancyArrowPatch((5.8, 7.2), (7.2, 5.8),
                            arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    ax1.add_patch(arrow2)
    
    # Factor to Returns (true causal effect)
    arrow3 = FancyArrowPatch((3.2, 5), (6.8, 5),
                            arrowstyle='->', mutation_scale=20, color='green', linewidth=3)
    ax1.add_patch(arrow3)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Subplot 2: Misspecified Model (Confounder Omitted)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('Misspecified Model\n(Confounder Omitted)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw nodes (only factor and returns visible in model)
    factor_circle2 = Circle((2, 5), 1.2, color=factor_color, alpha=0.8)
    return_circle2 = Circle((8, 5), 1.2, color=return_color, alpha=0.8)
    confounder_circle2 = Circle((5, 8), 1.2, color=confounder_color, alpha=0.3)  # Faded
    
    ax2.add_patch(factor_circle2)
    ax2.add_patch(return_circle2)
    ax2.add_patch(confounder_circle2)
    
    ax2.text(2, 5, 'Book-to-\nMarket', ha='center', va='center', fontweight='bold', fontsize=10)
    ax2.text(8, 5, 'Returns', ha='center', va='center', fontweight='bold', fontsize=10)
    ax2.text(5, 8, 'Leverage\n(Ignored)', ha='center', va='center', fontweight='bold', fontsize=10, alpha=0.5)
    
    # Spurious correlation (biased estimate)
    arrow_spurious = FancyArrowPatch((3.2, 5), (6.8, 5),
                                   arrowstyle='->', mutation_scale=20, color=wrong_color, 
                                   linewidth=3, linestyle='--')
    ax2.add_patch(arrow_spurious)
    
    # Hidden confounding paths (faded)
    arrow_hidden1 = FancyArrowPatch((4.2, 7.2), (2.8, 5.8),
                                  arrowstyle='->', mutation_scale=15, color='gray', 
                                  linewidth=1, alpha=0.5, linestyle=':')
    arrow_hidden2 = FancyArrowPatch((5.8, 7.2), (7.2, 5.8),
                                  arrowstyle='->', mutation_scale=15, color='gray', 
                                  linewidth=1, alpha=0.5, linestyle=':')
    ax2.add_patch(arrow_hidden1)
    ax2.add_patch(arrow_hidden2)
    
    # Add warning symbol
    ax2.text(5, 2, '⚠️ BIASED ESTIMATE', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=wrong_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=wrong_color))
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Subplot 3: Correct Model (Confounder Controlled)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_title('Correct Model\n(Confounder Controlled)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw all nodes
    factor_circle3 = Circle((2, 5), 1.2, color=factor_color, alpha=0.8)
    return_circle3 = Circle((8, 5), 1.2, color=return_color, alpha=0.8)
    confounder_circle3 = Circle((5, 8), 1.2, color=confounder_color, alpha=0.8)
    
    ax3.add_patch(factor_circle3)
    ax3.add_patch(return_circle3)
    ax3.add_patch(confounder_circle3)
    
    ax3.text(2, 5, 'Book-to-\nMarket', ha='center', va='center', fontweight='bold', fontsize=10)
    ax3.text(8, 5, 'Returns', ha='center', va='center', fontweight='bold', fontsize=10)
    ax3.text(5, 8, 'Leverage\n(Controlled)', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # All causal paths
    arrow_conf_factor = FancyArrowPatch((4.2, 7.2), (2.8, 5.8),
                                      arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    arrow_conf_return = FancyArrowPatch((5.8, 7.2), (7.2, 5.8),
                                      arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    arrow_true_effect = FancyArrowPatch((3.2, 5), (6.8, 5),
                                      arrowstyle='->', mutation_scale=20, color=correct_color, linewidth=3)
    
    ax3.add_patch(arrow_conf_factor)
    ax3.add_patch(arrow_conf_return)
    ax3.add_patch(arrow_true_effect)
    
    # Add checkmark
    ax3.text(5, 2, '✓ UNBIASED ESTIMATE', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=correct_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=correct_color))
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # Subplot 4: Confounder Bias Explanation
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Confounder Bias\nExplanation', fontsize=14, fontweight='bold', pad=20)
    
    explanation_text = """
CONFOUNDER BIAS occurs when:

• A variable influences BOTH the factor 
  and the outcome (returns)

• This creates a "backdoor path" that 
  confuses correlation with causation

• Example: Leverage affects both 
  book-to-market ratios AND returns

SOLUTION:
✓ Control for confounders in regression
✓ Use causal discovery methods
✓ Apply domain knowledge

LÓPEZ DE PRADO'S WARNING:
"Failing to control for confounders 
leads to biased estimates that can 
be wrong in both magnitude and sign"
    """
    
    ax4.text(0.5, 0.5, explanation_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF8E1', alpha=0.8))
    
    # ============================================================================
    # MIDDLE ROW: COLLIDER BIAS EXPLANATION
    # ============================================================================
    
    # Subplot 5: True Causal Structure (No Direct Relationship)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.set_title('True Causal Structure\n(Independent Variables)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw nodes
    factor_circle5 = Circle((2, 7), 1.2, color=factor_color, alpha=0.8)
    return_circle5 = Circle((8, 7), 1.2, color=return_color, alpha=0.8)
    collider_circle5 = Circle((5, 3), 1.2, color=collider_color, alpha=0.8)
    
    ax5.add_patch(factor_circle5)
    ax5.add_patch(return_circle5)
    ax5.add_patch(collider_circle5)
    
    ax5.text(2, 7, 'Book-to-\nMarket', ha='center', va='center', fontweight='bold', fontsize=10)
    ax5.text(8, 7, 'Returns', ha='center', va='center', fontweight='bold', fontsize=10)
    ax5.text(5, 3, 'Quality\n(Collider)', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Arrows pointing TO collider
    arrow_factor_collider = FancyArrowPatch((2.8, 6.2), (4.2, 3.8),
                                          arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    arrow_return_collider = FancyArrowPatch((7.2, 6.2), (5.8, 3.8),
                                          arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    
    ax5.add_patch(arrow_factor_collider)
    ax5.add_patch(arrow_return_collider)
    
    # No direct relationship between factor and returns
    ax5.text(5, 8.5, 'NO DIRECT CAUSAL\nRELATIONSHIP', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='gray',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))
    
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    
    # Subplot 6: Misspecified Model (Collider Controlled)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.set_title('Misspecified Model\n(Collider Controlled)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw nodes
    factor_circle6 = Circle((2, 7), 1.2, color=factor_color, alpha=0.8)
    return_circle6 = Circle((8, 7), 1.2, color=return_color, alpha=0.8)
    collider_circle6 = Circle((5, 3), 1.2, color=collider_color, alpha=0.8)
    
    ax6.add_patch(factor_circle6)
    ax6.add_patch(return_circle6)
    ax6.add_patch(collider_circle6)
    
    ax6.text(2, 7, 'Book-to-\nMarket', ha='center', va='center', fontweight='bold', fontsize=10)
    ax6.text(8, 7, 'Returns', ha='center', va='center', fontweight='bold', fontsize=10)
    ax6.text(5, 3, 'Quality\n(Controlled)', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Original causal arrows
    arrow_factor_collider6 = FancyArrowPatch((2.8, 6.2), (4.2, 3.8),
                                           arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    arrow_return_collider6 = FancyArrowPatch((7.2, 6.2), (5.8, 3.8),
                                           arrowstyle='->', mutation_scale=20, color='black', linewidth=2)
    
    ax6.add_patch(arrow_factor_collider6)
    ax6.add_patch(arrow_return_collider6)
    
    # Spurious correlation induced by controlling for collider
    arrow_spurious6 = FancyArrowPatch((3.2, 7), (6.8, 7),
                                    arrowstyle='<->', mutation_scale=20, color=wrong_color, 
                                    linewidth=3, linestyle='--')
    ax6.add_patch(arrow_spurious6)
    
    ax6.text(5, 8.5, 'SPURIOUS CORRELATION\nINDUCED!', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=wrong_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=wrong_color))
    
    # Add warning
    ax6.text(5, 0.5, '⚠️ COLLIDER BIAS', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=wrong_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=wrong_color))
    
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    
    # Subplot 7: Correct Model (Collider Not Controlled)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    ax7.set_title('Correct Model\n(Collider Not Controlled)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw nodes
    factor_circle7 = Circle((2, 7), 1.2, color=factor_color, alpha=0.8)
    return_circle7 = Circle((8, 7), 1.2, color=return_color, alpha=0.8)
    collider_circle7 = Circle((5, 3), 1.2, color=collider_color, alpha=0.5)  # Faded
    
    ax7.add_patch(factor_circle7)
    ax7.add_patch(return_circle7)
    ax7.add_patch(collider_circle7)
    
    ax7.text(2, 7, 'Book-to-\nMarket', ha='center', va='center', fontweight='bold', fontsize=10)
    ax7.text(8, 7, 'Returns', ha='center', va='center', fontweight='bold', fontsize=10)
    ax7.text(5, 3, 'Quality\n(Ignored)', ha='center', va='center', fontweight='bold', fontsize=10, alpha=0.7)
    
    # Faded arrows to show collider exists but isn't controlled
    arrow_factor_collider7 = FancyArrowPatch((2.8, 6.2), (4.2, 3.8),
                                           arrowstyle='->', mutation_scale=15, color='gray', 
                                           linewidth=1, alpha=0.5, linestyle=':')
    arrow_return_collider7 = FancyArrowPatch((7.2, 6.2), (5.8, 3.8),
                                           arrowstyle='->', mutation_scale=15, color='gray', 
                                           linewidth=1, alpha=0.5, linestyle=':')
    
    ax7.add_patch(arrow_factor_collider7)
    ax7.add_patch(arrow_return_collider7)
    
    ax7.text(5, 8.5, 'TRUE RELATIONSHIP\nPRESERVED', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=correct_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=correct_color))
    
    # Add checkmark
    ax7.text(5, 0.5, '✓ NO BIAS', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=correct_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=correct_color))
    
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.spines['bottom'].set_visible(False)
    ax7.spines['left'].set_visible(False)
    
    # Subplot 8: Collider Bias Explanation
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_xlim(0, 10)
    ax8.set_ylim(0, 10)
    ax8.axis('off')
    ax8.set_title('Collider Bias\nExplanation', fontsize=14, fontweight='bold', pad=20)
    
    explanation_text2 = """
COLLIDER BIAS occurs when:

• A variable is influenced by BOTH 
  the factor and the outcome

• Controlling for it creates a spurious 
  correlation where none exists

• Example: Quality is affected by both 
  book-to-market AND returns

DANGER SIGNS:
⚠️ R-squared increases when controlling
⚠️ Coefficient signs flip
⚠️ Standard errors decrease (misleading!)

LÓPEZ DE PRADO'S WARNING:
"Colliders are particularly dangerous 
because they often change the sign of 
estimated coefficients, inducing 
investors to buy securities that should 
be sold, and vice versa"
    """
    
    ax8.text(0.5, 0.5, explanation_text2, transform=ax8.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFEBEE', alpha=0.8))
    
    # ============================================================================
    # BOTTOM ROW: PRACTICAL IMPLICATIONS AND SUMMARY
    # ============================================================================
    
    # Large subplot for practical implications
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    ax9.set_title('Practical Implications for Factor Investing', fontsize=16, fontweight='bold', pad=20)
    
    # Create boxes for different sections
    box_width = 0.22
    box_height = 0.8
    y_pos = 0.1
    
    # Box 1: Detection Methods
    detection_box = FancyBboxPatch((0.02, y_pos), box_width, box_height,
                                 boxstyle="round,pad=0.02", 
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax9.add_patch(detection_box)
    
    detection_text = """DETECTION METHODS

• Test R² changes when adding controls
• Check for coefficient sign flips  
• Use causal discovery algorithms
• Apply domain knowledge
• Monitor first-stage F-statistics
• Examine partial correlations

KEY INSIGHT:
Traditional metrics (R², t-stats) 
can reward misspecification!"""
    
    ax9.text(0.03, y_pos + 0.02, detection_text, fontsize=10, verticalalignment='bottom',
            fontweight='normal', transform=ax9.transAxes)
    
    # Box 2: Factor Examples
    examples_box = FancyBboxPatch((0.26, y_pos), box_width, box_height,
                                boxstyle="round,pad=0.02", 
                                facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax9.add_patch(examples_box)
    
    examples_text = """FACTOR EXAMPLES

CONFOUNDERS (Control for):
• Leverage → B/M & Returns
• Size → Multiple factors  
• Industry → Factor exposure
• Market regime → All factors

COLLIDERS (Don't control):
• Quality ← B/M & Returns
• Momentum ← Size & Returns  
• Volatility ← Value & Returns
• Analyst coverage ← Multiple"""
    
    ax9.text(0.27, y_pos + 0.02, examples_text, fontsize=10, verticalalignment='bottom',
            fontweight='normal', transform=ax9.transAxes)
    
    # Box 3: Investment Consequences  
    consequences_box = FancyBboxPatch((0.50, y_pos), box_width, box_height,
                                    boxstyle="round,pad=0.02", 
                                    facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=2)
    ax9.add_patch(consequences_box)
    
    consequences_text = """INVESTMENT CONSEQUENCES

CONFOUNDER BIAS:
• Overestimate factor premiums
• Attribute returns to wrong factors
• Poor out-of-sample performance
• Excessive portfolio turnover

COLLIDER BIAS:
• Create spurious factor relationships
• Flip investment signals (buy→sell)
• Inflate backtested Sharpe ratios
• Systematic portfolio losses"""
    
    ax9.text(0.51, y_pos + 0.02, consequences_text, fontsize=10, verticalalignment='bottom',
            fontweight='normal', transform=ax9.transAxes)
    
    # Box 4: Solutions
    solutions_box = FancyBboxPatch((0.74, y_pos), box_width, box_height,
                                 boxstyle="round,pad=0.02", 
                                 facecolor='#E8F5E8', edgecolor='#388E3C', linewidth=2)
    ax9.add_patch(solutions_box)
    
    solutions_text = """LÓPEZ DE PRADO'S SOLUTION

7-STEP PROTOCOL:
1. Variable selection (ML methods)
2. Causal discovery (PC, LiNGAM)  
3. Causal adjustment (do-calculus)
4. Causal estimation (Double ML)
5. Portfolio construction
6. Robust backtesting
7. Multiple testing correction

GOAL: Move from associational 
to causal factor investing"""
    
    ax9.text(0.75, y_pos + 0.02, solutions_text, fontsize=10, verticalalignment='bottom',
            fontweight='normal', transform=ax9.transAxes)
    
    # Add footer with citation
    footer_text = ('Source: López de Prado, M. & Zoonekynd, V. (2025). "A Protocol for Causal Factor Investing." ADIA Lab Research Series.\n'
                  'Implementation: Enhanced Factor Mirage Detection for Causal Discovery in Factor Investing')
    
    ax9.text(0.5, -0.05, footer_text, transform=ax9.transAxes, fontsize=9,
            ha='center', va='top', style='italic', color='gray')
    
    plt.tight_layout()
    return fig

def save_graph():
    """Save the confounder and collider bias explanation graph"""
    
    # Create the graph
    fig = create_confounder_collider_explanation()
    
    # Save with high DPI for thesis quality
    output_path = '../Synthetic/confounder_collider_bias_explanation.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Confounder and Collider Bias Explanation graph saved to: {output_path}")
    print("\nGraph Features:")
    print("• Comprehensive visual explanation of both bias types")
    print("• Factor investing examples (book-to-market, leverage, quality)")
    print("• López de Prado's causal framework integration")
    print("• Practical detection methods and solutions")
    print("• Academic formatting")
    
    plt.show()
    
    return output_path

if __name__ == "__main__":
    save_graph() 