#!/usr/bin/env python3
"""
Generate Clean Confounder and Collider Bias Explanation
Professional academic-style visualization for master's thesis

Based on López de Prado's "A Protocol for Causal Factor Investing"
Creates clean, focused diagrams explaining confounder and collider bias
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import matplotlib.patheffects as path_effects

# Set academic style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'

def create_clean_confounder_explanation():
    """Create clean confounder bias explanation"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'CONFOUNDER BIAS IN FACTOR INVESTING', 
            fontsize=18, fontweight='bold', ha='center', va='center')
    
    # Colors
    factor_color = '#2E86AB'      # Blue
    confounder_color = '#F18F01'  # Orange  
    return_color = '#A23B72'      # Purple
    path_color = '#E74C3C'        # Red
    
    # Draw main boxes
    # Factor box
    factor_box = FancyBboxPatch((0.5, 4), 2, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=factor_color, edgecolor='black', linewidth=2)
    ax.add_patch(factor_box)
    ax.text(1.5, 4.6, 'Book-to-Market\nFactor', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    
    # Confounder box
    conf_box = FancyBboxPatch((4, 5.5), 2, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=confounder_color, edgecolor='black', linewidth=2)
    ax.add_patch(conf_box)
    ax.text(5, 6.1, 'Leverage\n(Confounder)', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    
    # Returns box
    returns_box = FancyBboxPatch((7.5, 4), 2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=return_color, edgecolor='black', linewidth=2)
    ax.add_patch(returns_box)
    ax.text(8.5, 4.6, 'Stock\nReturns', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    
    # Draw arrows
    # Direct causal effect
    ax.annotate('', xy=(7.3, 4.6), xytext=(2.7, 4.6),
                arrowprops=dict(arrowstyle='->', lw=3, color=factor_color))
    ax.text(5, 5, 'Direct Causal Effect\n(+2.1% per σ)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=factor_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=factor_color))
    
    # Confounding paths
    ax.annotate('', xy=(4.2, 5.8), xytext=(2.3, 5.0),
                arrowprops=dict(arrowstyle='->', lw=2, color=path_color, linestyle='--'))
    ax.annotate('', xy=(7.7, 5.0), xytext=(5.8, 5.8),
                arrowprops=dict(arrowstyle='->', lw=2, color=path_color, linestyle='--'))
    
    # Confounding path label
    ax.text(5, 6.8, 'CONFOUNDING PATH', ha='center', va='center',
            fontsize=11, fontweight='bold', color=path_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=path_color))
    
    # Selection bias explanation
    ax.text(1.5, 3.2, 'Selection Bias:\nHigh leverage →\nLower B/M ratios', 
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFE5CC', alpha=0.8))
    
    # Problem explanation box
    problem_box = FancyBboxPatch((0.2, 0.3), 4.5, 2.2,
                                boxstyle="round,pad=0.15",
                                facecolor='#FFEBEE', edgecolor='#E74C3C', linewidth=2)
    ax.add_patch(problem_box)
    
    problem_text = """CONFOUNDING PROBLEM:
• Naïve comparison: Value vs. Growth returns
• Problem: Value stocks often have different leverage
• Leverage affects both B/M ratios and returns directly  
• Simple comparison conflates factor and leverage effects
• Leads to biased factor premium estimates"""
    
    ax.text(0.4, 1.4, problem_text, fontsize=10, va='center', ha='left', fontweight='normal')
    
    # Solution explanation box
    solution_box = FancyBboxPatch((5.3, 0.3), 4.3, 2.2,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#E8F5E8', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(solution_box)
    
    solution_text = """CAUSAL INFERENCE SOLUTIONS:
• Control for leverage in regression models
• Matching: Compare similar leverage stocks
• Instrumental Variables: Use exogenous variation
• Optimal Transport: Minimize distributional imbalances
• Double ML: Flexibly estimate confounding effects"""
    
    ax.text(5.5, 1.4, solution_text, fontsize=10, va='center', ha='left', fontweight='normal')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=factor_color, lw=3, label='Direct Factor Effect'),
        plt.Line2D([0], [0], color=path_color, lw=2, linestyle='--', label='Confounding Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_clean_collider_explanation():
    """Create clean collider bias explanation"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'COLLIDER BIAS IN FACTOR INVESTING', 
            fontsize=18, fontweight='bold', ha='center', va='center')
    
    # Colors
    factor_color = '#2E86AB'      # Blue
    collider_color = '#E74C3C'    # Red
    return_color = '#A23B72'      # Purple
    spurious_color = '#FF6B35'    # Orange-red
    
    # Draw main boxes
    # Factor box
    factor_box = FancyBboxPatch((0.5, 5.5), 2, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=factor_color, edgecolor='black', linewidth=2)
    ax.add_patch(factor_box)
    ax.text(1.5, 6.1, 'Book-to-Market\nFactor', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    
    # Returns box
    returns_box = FancyBboxPatch((7.5, 5.5), 2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=return_color, edgecolor='black', linewidth=2)
    ax.add_patch(returns_box)
    ax.text(8.5, 6.1, 'Stock\nReturns', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    
    # Collider box
    collider_box = FancyBboxPatch((4, 3.5), 2, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=collider_color, edgecolor='black', linewidth=2)
    ax.add_patch(collider_box)
    ax.text(5, 4.1, 'Quality Score\n(Collider)', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    
    # Draw arrows pointing TO collider
    ax.annotate('', xy=(4.2, 4.4), xytext=(2.3, 5.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(5.8, 4.4), xytext=(7.7, 5.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # No direct relationship (crossed out)
    ax.plot([2.5, 7.5], [6.1, 6.1], 'k--', alpha=0.3, linewidth=2)
    ax.text(5, 6.5, 'NO DIRECT CAUSAL RELATIONSHIP', ha='center', va='center',
            fontsize=10, fontweight='bold', color='gray', style='italic')
    
    # Spurious correlation when conditioning on collider
    ax.annotate('', xy=(7.3, 5.8), xytext=(2.7, 5.8),
                arrowprops=dict(arrowstyle='<->', lw=3, color=spurious_color, linestyle=':'))
    ax.text(5, 5.2, 'Spurious Correlation\n(when controlling for Quality)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color=spurious_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=spurious_color))
    
    # Collider label
    ax.text(5, 2.8, 'COLLIDER VARIABLE', ha='center', va='center',
            fontsize=11, fontweight='bold', color=collider_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=collider_color))
    
    # Problem explanation box
    problem_box = FancyBboxPatch((0.2, 0.3), 4.5, 2.2,
                                boxstyle="round,pad=0.15",
                                facecolor='#FFEBEE', edgecolor='#E74C3C', linewidth=2)
    ax.add_patch(problem_box)
    
    problem_text = """COLLIDER BIAS PROBLEM:
• Quality is influenced by BOTH B/M and returns
• Controlling for quality opens "collider path"
• Creates spurious correlation between B/M and returns
• Can flip coefficient signs (dangerous!)
• Inflates R² and significance (misleading)"""
    
    ax.text(0.4, 1.4, problem_text, fontsize=10, va='center', ha='left', fontweight='normal')
    
    # Solution explanation box
    solution_box = FancyBboxPatch((5.3, 0.3), 4.3, 2.2,
                                 boxstyle="round,pad=0.15",
                                 facecolor='#E8F5E8', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(solution_box)
    
    solution_text = """AVOIDING COLLIDER BIAS:
• DON'T control for variables caused by treatment
• Use causal graphs to identify colliders
• Test for coefficient sign changes
• Monitor R² increases (warning sign)
• Apply López de Prado's factor mirage detection"""
    
    ax.text(5.5, 1.4, solution_text, fontsize=10, va='center', ha='left', fontweight='normal')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=2, label='Causal Effects'),
        plt.Line2D([0], [0], color=spurious_color, lw=3, linestyle=':', label='Spurious Correlation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_combined_clean_diagram():
    """Create a combined clean diagram showing both biases side by side with enhanced detail"""
    
    fig = plt.figure(figsize=(12, 10))
    
    # Create two subplots with adjusted spacing
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Adjust subplot positions to move bottom graph up
    plt.subplots_adjust(hspace=0.3)
    
    # CONFOUNDER BIAS (TOP)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 6.5)
    ax1.axis('off')
    ax1.text(6, 6, 'CONFOUNDER BIAS', fontsize=16, fontweight='bold', ha='center')
    
    # Colors
    factor_color = '#2E86AB'
    confounder_color = '#F18F01'
    return_color = '#A23B72'
    path_color = '#E74C3C'
    
    # Confounder diagram with enhanced detail - bigger boxes
    factor_box1 = FancyBboxPatch((0.8, 2.8), 2.4, 1, boxstyle="round,pad=0.08", 
                                facecolor=factor_color, edgecolor='black', linewidth=2)
    ax1.add_patch(factor_box1)
    ax1.text(2, 3.3, 'Factor X\n(Treatment)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    conf_box1 = FancyBboxPatch((4.8, 4.3), 2.4, 1, boxstyle="round,pad=0.08",
                              facecolor=confounder_color, edgecolor='black', linewidth=2)
    ax1.add_patch(conf_box1)
    ax1.text(6, 4.8, 'Variable Z\n(Confounder)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    returns_box1 = FancyBboxPatch((8.8, 2.8), 2.4, 1, boxstyle="round,pad=0.08",
                                 facecolor=return_color, edgecolor='black', linewidth=2)
    ax1.add_patch(returns_box1)
    ax1.text(10, 3.3, 'Outcome Y\n(What we measure)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    # Enhanced arrows with labels
    # Direct causal effect
    ax1.annotate('', xy=(8.6, 3.3), xytext=(3.4, 3.3),
                arrowprops=dict(arrowstyle='->', lw=3, color=factor_color))
    ax1.text(6, 3.7, 'True Causal Effect', ha='center', va='center',
             fontsize=11, fontweight='bold', color=factor_color,
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=factor_color, alpha=0.9))
    
    # Confounding paths with detailed labels
    ax1.annotate('', xy=(4.9, 4.4), xytext=(3.1, 3.6),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=path_color, linestyle='--'))
    ax1.annotate('', xy=(9.0, 3.6), xytext=(7.1, 4.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=path_color, linestyle='--'))
    
    # Simplified mechanism explanations - better positioned
    ax1.text(3.8, 5.2, 'Z affects\nX values', 
             ha='center', va='center', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFE5CC', alpha=0.8))
    
    ax1.text(8.2, 5.2, 'Z also affects\nY directly', 
             ha='center', va='center', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFE5CC', alpha=0.8))
    
    # Backdoor path illustration
    ax1.text(6, 2.5, 'CONFOUNDING PATH', ha='center', va='center',
             fontsize=11, fontweight='bold', color=path_color,
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=path_color))
    
    # Conceptual explanation
    ax1.text(6, 1.7, 'Problem: Without controlling for Z, we get a biased estimate\nof the true X → Y effect (mixing up causes)', 
             ha='center', fontsize=10, color='black', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='#F0F8FF', alpha=0.8))
    
    ax1.text(6, 1.1, 'Solution: Control for Z to isolate the true X → Y effect', 
             ha='center', fontsize=10, color='#4CAF50', fontweight='bold')
    
    # COLLIDER BIAS (BOTTOM)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6.5)
    ax2.axis('off')
    ax2.text(6, 6, 'COLLIDER BIAS', fontsize=16, fontweight='bold', ha='center')
    
    collider_color = '#E74C3C'
    spurious_color = '#FF6B35'
    
    factor_box2 = FancyBboxPatch((0.8, 3.8), 2.4, 1, boxstyle="round,pad=0.08", 
                                facecolor=factor_color, edgecolor='black', linewidth=2)
    ax2.add_patch(factor_box2)
    ax2.text(2, 4.3, 'Factor X\n(Treatment)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    returns_box2 = FancyBboxPatch((8.8, 3.8), 2.4, 1, boxstyle="round,pad=0.08",
                                 facecolor=return_color, edgecolor='black', linewidth=2)
    ax2.add_patch(returns_box2)
    ax2.text(10, 4.3, 'Outcome Y\n(What we measure)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    collider_box2 = FancyBboxPatch((4.8, 2), 2.4, 1, boxstyle="round,pad=0.08",
                                  facecolor=collider_color, edgecolor='black', linewidth=2)
    ax2.add_patch(collider_box2)
    ax2.text(6, 2.5, 'Variable W\n(Collider)', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    # Enhanced arrows TO collider with detailed labels
    ax2.annotate('', xy=(4.9, 2.7), xytext=(3.1, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax2.annotate('', xy=(7.1, 2.7), xytext=(9.0, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    
    # Simplified mechanism explanations - better positioned
    ax2.text(3.8, 3.2, 'X affects\nW values', 
             ha='center', va='center', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFE5CC', alpha=0.8))
    
    ax2.text(8.2, 3.2, 'Y also affects\nW values', 
             ha='center', va='center', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFE5CC', alpha=0.8))
    
    # No direct relationship emphasized
    ax2.plot([3.2, 8.8], [4.3, 4.3], 'k--', alpha=0.4, linewidth=3)
    ax2.text(6, 5.2, 'TRUE: No Direct X → Y Relationship', ha='center', va='center',
             fontsize=11, fontweight='bold', color='gray', style='italic',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.8))
    
    # Spurious correlation with simplified warning
    ax2.annotate('', xy=(8.6, 4.1), xytext=(3.4, 4.1),
                arrowprops=dict(arrowstyle='<->', lw=3, color=spurious_color, linestyle=':'))
    
    ax2.text(6, 3.7, 'SPURIOUS CORRELATION\n(when controlling for W)\nFalse relationship appears!', 
             ha='center', va='center', fontsize=10, fontweight='bold', color=spurious_color,
             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=spurious_color, alpha=0.9))
    
    # Conceptual warning
    ax2.text(6, 1.7, 'Problem: Controlling for W creates a false relationship\nbetween X and Y (where none exists)', 
             ha='center', fontsize=10, color=spurious_color, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFEBEE', alpha=0.8))
    
    ax2.text(6, 1.1, 'Solution: Do NOT control for variables that are effects of both X and Y', 
             ha='center', fontsize=10, color='#4CAF50', fontweight='bold')
    
    # No citation in figure - handled through bibliography
    
    plt.tight_layout()
    return fig

def save_clean_graphs():
    """Save the clean professional graphs"""
    
    # Create and save combined diagram
    fig_combined = create_combined_clean_diagram()
    output_path = '../Synthetic/confounder_collider_bias_explanation.png'
    
    fig_combined.savefig(output_path, dpi=400, bbox_inches='tight', 
                        facecolor='white', edgecolor='none', format='png')
    
    print(f"✅ Clean Confounder and Collider Bias diagram saved to: {output_path}")
    print("\n📊 Professional Features:")
    print("• Clean, academic-style visualization")
    print("• Clear causal diagrams with proper arrows")
    print("• Factor investing examples (B/M, leverage, quality)")
    print("• Concise problem and solution explanations")
    print("• Master's thesis quality formatting")
    print("• López de Prado's methodology integration")
    
    plt.show()
    return output_path

if __name__ == "__main__":
    save_clean_graphs() 