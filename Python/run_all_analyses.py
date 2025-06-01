#!/usr/bin/env python3
"""
Main runner script to execute all analyses and generate visualizations.

This script runs both synthetic and real data analyses, including:
- Synthetic data generation and analysis
- Real Fama-French data analysis
- All visualization generation

Usage:
    python run_all_analyses.py [--synthetic-only] [--real-only] [--graphs-only]
"""

import os
import sys
import argparse
import subprocess

def run_python_file(filepath, description):
    """Run a Python file and handle any errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {filepath}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, filepath], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {description}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Main function to coordinate all analyses"""
    parser = argparse.ArgumentParser(description='Run factor investing causal analyses')
    parser.add_argument('--synthetic-only', action='store_true', 
                       help='Run only synthetic data analysis')
    parser.add_argument('--real-only', action='store_true', 
                       help='Run only real data analysis')
    parser.add_argument('--graphs-only', action='store_true', 
                       help='Generate only graphs (skip analyses)')
    
    args = parser.parse_args()
    
    # Determine what to run
    run_synthetic = not args.real_only
    run_real = not args.synthetic_only
    run_analyses = not args.graphs_only
    
    print("FACTOR INVESTING CAUSAL ANALYSIS RUNNER")
    print("="*60)
    print(f"Run synthetic data: {run_synthetic}")
    print(f"Run real data: {run_real}")
    print(f"Run analyses: {run_analyses}")
    
    success = True
    
    # Run synthetic data analysis
    if run_synthetic:
        if run_analyses:
            success &= run_python_file(
                "Python/Analysis/Causality_Main.py",
                "Synthetic Data Analysis"
            )
        
        success &= run_python_file(
            "Python/Visualization/Causality_Main_Graphs.py",
            "Synthetic Data Visualization"
        )
    
    # Run real data analysis
    if run_real:
        if run_analyses:
            success &= run_python_file(
                "Python/Analysis/Causality_Real_Data.py",
                "Real Data Analysis (Fama-French)"
            )
        
        success &= run_python_file(
            "Python/Visualization/Causality_Real_Data_Graphs.py",
            "Real Data Visualization"
        )
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    if success:
        print("✓ All analyses completed successfully!")
        
        if run_synthetic:
            print(f"\nSynthetic data graphs saved to: Graphs/Synthetic/")
        if run_real:
            print(f"Real data graphs saved to: Graphs/Real Data/")
            
        print("\nYou can find:")
        print("- Analysis scripts in: Python/Analysis/")
        print("- Visualization scripts in: Python/Visualization/")
        print("- LaTeX thesis files in: Overleaf/")
    else:
        print("✗ Some analyses failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 