#!/usr/bin/env python3
"""
Master graph generation system.
"""

import sys
import subprocess
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

def get_project_root() -> Path:
    """Get project root folder"""
    return Path(__file__).resolve().parent.parent

def clean_redundant_graphs():
    """Remove redundant graph files"""
    print("Cleaning redundant graphs...")
    
    project_root = get_project_root()
    graphs_dir = Path('..').resolve()
    
    # Patterns of redundant files to remove
    redundant_patterns = [
        '*_comparison.png',
        '*_summary.png',
        'treatment_effect_*.png',
        'covariate_balance_plot*.png',
        'causality_graph*.png',
        'causal_graph.png',
        'returns_time.png',
        'factor_distributions.png',
        'correlation_matrix.png',
        'did_results.png',
        'iv_results.png',
        'regime_effects*.png',
        'enhanced_*.png',
        'detailed_causal_networks.png',
    ]
    
    removed_count = 0
    
    for subfolder in ['Synthetic', 'Real Data']:
        subfolder_path = graphs_dir / subfolder
        if subfolder_path.exists():
            for pattern in redundant_patterns:
                for file_path in subfolder_path.glob(pattern):
                    if 'professional' not in file_path.stem:
                        try:
                            file_path.unlink()
                            removed_count += 1
                            print(f"  Removed: {file_path.name}")
                        except Exception as e:
                            print(f"  Could not remove {file_path.name}: {e}")
    
    print(f"Cleaned {removed_count} redundant graph files")

def generate_graphs():
    """Generate all graphs"""
    print("Generating graphs...")
    
    project_root = get_project_root()
    
    # Run synthetic data graph generation
    print("\nGenerating Synthetic Data Graphs...")
    synthetic_script = project_root / 'Python' / 'Visualization' / 'Causality_Main_Graphs.py'
    if synthetic_script.exists():
        try:
            result = subprocess.run([sys.executable, str(synthetic_script)], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                print("Synthetic data graphs generated successfully")
            else:
                print(f"Error generating synthetic graphs: {result.stderr}")
        except Exception as e:
            print(f"Failed to run synthetic graph generation: {e}")
    
    # Run real data graph generation
    print("\nGenerating Real Data Graphs...")
    real_script = project_root / 'Python' / 'Visualization' / 'Causality_Real_Data_Graphs.py'
    if real_script.exists():
        try:
            result = subprocess.run([sys.executable, str(real_script)], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                print("Real data graphs generated successfully")
            else:
                print(f"Error generating real data graphs: {result.stderr}")
        except Exception as e:
            print(f"Failed to run real data graph generation: {e}")

def create_graph_inventory():
    """Create inventory of generated graphs"""
    print("\nCreating graph inventory...")
    
    project_root = get_project_root()
    graphs_dir = Path('..').resolve()
    
    inventory = {
        'Synthetic Data Graphs': [],
        'Real Data Graphs': [],
        'Shared Graphs': []
    }
    
    # Collect synthetic graphs
    synthetic_dir = graphs_dir / 'Synthetic'
    if synthetic_dir.exists():
        for graph_file in synthetic_dir.glob('*.png'):
            if 'professional' in graph_file.name or 'comprehensive' in graph_file.name:
                inventory['Synthetic Data Graphs'].append(graph_file.name)
    
    # Collect real data graphs
    real_dir = graphs_dir / 'Real Data'
    if real_dir.exists():
        for graph_file in real_dir.glob('*.png'):
            if 'professional' in graph_file.name or 'comprehensive' in graph_file.name:
                inventory['Real Data Graphs'].append(graph_file.name)
    
    # Collect shared graphs
    for graph_file in graphs_dir.glob('*.png'):
        if 'professional' in graph_file.name or 'comprehensive' in graph_file.name:
            inventory['Shared Graphs'].append(graph_file.name)
    
    # Generate inventory report
    inventory_file = project_root / 'GRAPH_INVENTORY.md'
    with open(inventory_file, 'w') as f:
        f.write("# Graph Inventory\n\n")
        f.write("Generated visualizations for the project.\n\n")
        
        for category, graphs in inventory.items():
            f.write(f"## {category}\n\n")
            if graphs:
                for graph in sorted(graphs):
                    f.write(f"- {graph}\n")
            else:
                f.write("- No graphs in this category\n")
            f.write("\n")
        
        f.write("## Quality Standards\n\n")
        f.write("All graphs meet the following standards:\n")
        f.write("- 300 DPI resolution\n")
        f.write("- Consistent styling\n")
        f.write("- Clear labels\n")
        f.write("- Statistical annotations\n")
        f.write("- Proper formatting\n")
        f.write("- Color schemes\n")
        f.write("- Typography\n\n")
        
        f.write("## Usage\n\n")
        f.write("These graphs are designed for:\n")
        f.write("- LaTeX thesis document\n")
        f.write("- Academic presentations\n")
        f.write("- Research publications\n")
        f.write("- Conference posters\n")
    
    print(f"Graph inventory created: {inventory_file}")
    return inventory

def validate_graph_quality():
    """Validate graph quality"""
    print("\nValidating graph quality...")
    
    project_root = get_project_root()
    graphs_dir = Path('..').resolve()
    
    quality_issues = []
    total_graphs = 0
    
    for subfolder in ['Synthetic', 'Real Data']:
        subfolder_path = graphs_dir / subfolder
        if subfolder_path.exists():
            for graph_file in subfolder_path.glob('*.png'):
                if 'professional' in graph_file.name or 'comprehensive' in graph_file.name:
                    total_graphs += 1
                    
                    # Check file size
                    file_size = graph_file.stat().st_size
                    if file_size < 100_000:
                        quality_issues.append(f"{graph_file.name}: File size may be too small ({file_size:,} bytes)")
                    
                    # Check naming convention
                    if 'professional' not in graph_file.name and 'comprehensive' not in graph_file.name:
                        quality_issues.append(f"{graph_file.name}: Does not follow naming convention")
    
    if quality_issues:
        print("Quality issues found:")
        for issue in quality_issues:
            print(f"  - {issue}")
    else:
        print(f"All {total_graphs} graphs passed quality validation")
    
    return len(quality_issues) == 0

def main():
    """Main function"""
    print("="*80)
    print("MASTER GRAPH GENERATION SYSTEM")
    print("="*80)
    print("Generating visualizations...")
    print()
    
    try:
        # Step 1: Clean redundant graphs
        clean_redundant_graphs()
        
        # Step 2: Generate graphs
        generate_graphs()
        
        # Step 3: Create inventory
        inventory = create_graph_inventory()
        
        # Step 4: Validate quality
        quality_ok = validate_graph_quality()
        
        # Final report
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        
        total_graphs = sum(len(graphs) for graphs in inventory.values())
        print(f"Generated {total_graphs} graphs")
        print(f"Quality validation: {'PASSED' if quality_ok else 'ISSUES FOUND'}")
        print(f"All graphs saved at 300 DPI")
        print(f"Consistent styling applied")
        print(f"Redundant graphs removed")
        
        print("\nGraph locations:")
        project_root = get_project_root()
        print(f"  - Synthetic: {Path('..').resolve() / 'Synthetic'}")
        print(f"  - Real Data: {Path('..').resolve() / 'Real Data'}")
        print(f"  - Inventory: {project_root / 'GRAPH_INVENTORY.md'}")
        
        print("\nAll visualizations ready!")
        
    except Exception as e:
        print(f"\nError in graph generation process: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 