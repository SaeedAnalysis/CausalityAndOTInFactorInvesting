#!/usr/bin/env python3
"""
Build metrics and summaries for causal discovery analysis.

This script runs the causal discovery algorithms on both synthetic and real data,
then compiles the results into a comprehensive metrics report.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def run_analysis_script(script_path, description):
    """
    Run an analysis script and capture key metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=str(Path(script_path).parent)
        )
        
        # Check for errors
        if result.returncode != 0:
            print(f"❌ Error running {script_path}")
            print(f"Error output: {result.stderr}")
            return {"status": "error", "error": result.stderr}
        
        # Extract metrics from output
        output_lines = result.stdout.split('\n')
        metrics = {
            "status": "success",
            "output_lines": len(output_lines)
        }
        
        # Look for accuracy metrics in output
        for line in output_lines:
            if "Accuracy:" in line and "%" in line:
                try:
                    accuracy = float(line.split("%")[0].split()[-1])
                    metrics["accuracy"] = accuracy
                except:
                    pass
            elif "PC Algorithm Accuracy:" in line:
                try:
                    accuracy = float(line.split("(")[0].split()[-1].strip("%"))
                    metrics["pc_accuracy"] = accuracy
                except:
                    pass
            elif "ANM Accuracy:" in line:
                try:
                    accuracy = float(line.split()[-1].strip("%"))
                    metrics["anm_accuracy"] = accuracy
                except:
                    pass
            elif "DIVOT Accuracy:" in line:
                try:
                    accuracy = float(line.split()[-1].strip("%"))
                    metrics["divot_accuracy"] = accuracy
                except:
                    pass
        
        print(f"✅ Successfully completed {description}")
        return metrics
        
    except Exception as e:
        print(f"❌ Exception running {script_path}: {str(e)}")
        return {"status": "error", "error": str(e)}

def build_comprehensive_metrics():
    """
    Build comprehensive metrics from all analyses.
    """
    print("\n" + "="*60)
    print("BUILDING COMPREHENSIVE METRICS")
    print("="*60)
    
    # Define project paths
    project_root = Path(__file__).parent.parent
    analysis_dir = project_root / "Python" / "Analysis"
    
    # Initialize metrics dictionary
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "analyses": {}
    }
    
    # Run synthetic data analysis
    synthetic_script = analysis_dir / "Causality_Main.py"
    if synthetic_script.exists():
        metrics["analyses"]["synthetic"] = run_analysis_script(
            synthetic_script,
            "Synthetic Data Causal Discovery Analysis"
        )
    
    # Run real data analysis
    real_script = analysis_dir / "Causality_Real_Data.py"
    if real_script.exists():
        metrics["analyses"]["real"] = run_analysis_script(
            real_script,
            "Real Data Causal Discovery Analysis"
        )
    
    # Compile summary statistics
    summary = {
        "total_analyses": len(metrics["analyses"]),
        "successful_analyses": sum(1 for a in metrics["analyses"].values() if a.get("status") == "success"),
        "causal_discovery_methods": ["PC Algorithm", "ANM", "DIVOT"],
        "key_findings": []
    }
    
    # Extract key findings
    if "synthetic" in metrics["analyses"] and metrics["analyses"]["synthetic"].get("status") == "success":
        synthetic_metrics = metrics["analyses"]["synthetic"]
        if "pc_accuracy" in synthetic_metrics:
            summary["key_findings"].append(f"PC Algorithm accuracy on synthetic data: {synthetic_metrics['pc_accuracy']}%")
        if "anm_accuracy" in synthetic_metrics:
            summary["key_findings"].append(f"ANM accuracy on synthetic data: {synthetic_metrics['anm_accuracy']}%")
        if "divot_accuracy" in synthetic_metrics:
            summary["key_findings"].append(f"DIVOT accuracy on synthetic data: {synthetic_metrics['divot_accuracy']}%")
    
    metrics["summary"] = summary
    
    # Save metrics to file
    metrics_file = project_root / "Python" / "latest_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: {metrics_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(f"Total analyses run: {summary['total_analyses']}")
    print(f"Successful analyses: {summary['successful_analyses']}")
    print(f"Causal discovery methods: {', '.join(summary['causal_discovery_methods'])}")
    
    if summary["key_findings"]:
        print("\nKey Findings:")
        for finding in summary["key_findings"]:
            print(f"  • {finding}")
    
    return metrics

def generate_markdown_report(metrics):
    """
    Generate a markdown report from the metrics.
    """
    print("\n" + "="*60)
    print("GENERATING MARKDOWN REPORT")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    report_file = project_root / "METRICS_REPORT.md"
    
    with open(report_file, 'w') as f:
        f.write("# Causal Discovery Analysis Metrics Report\n\n")
        f.write(f"Generated: {metrics['timestamp']}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the results of causal discovery analysis using three methods:\n")
        f.write("- **PC Algorithm**: Constraint-based causal graph discovery\n")
        f.write("- **ANM (Additive Noise Model)**: Pairwise causal direction testing\n")
        f.write("- **DIVOT**: Optimal transport-based causal discovery\n\n")
        
        f.write("## Analysis Results\n\n")
        
        # Synthetic data results
        if "synthetic" in metrics["analyses"]:
            f.write("### Synthetic Data Analysis\n\n")
            synthetic = metrics["analyses"]["synthetic"]
            if synthetic.get("status") == "success":
                f.write("✅ **Status**: Successful\n\n")
                if "pc_accuracy" in synthetic:
                    f.write(f"- PC Algorithm Accuracy: **{synthetic['pc_accuracy']}%**\n")
                if "anm_accuracy" in synthetic:
                    f.write(f"- ANM Accuracy: **{synthetic['anm_accuracy']}%**\n")
                if "divot_accuracy" in synthetic:
                    f.write(f"- DIVOT Accuracy: **{synthetic['divot_accuracy']}%**\n")
            else:
                f.write("❌ **Status**: Failed\n")
                f.write(f"Error: {synthetic.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        # Real data results
        if "real" in metrics["analyses"]:
            f.write("### Real Data Analysis\n\n")
            real = metrics["analyses"]["real"]
            if real.get("status") == "success":
                f.write("✅ **Status**: Successful\n\n")
                f.write("Analysis of Fama-French factors and market returns completed.\n")
            else:
                f.write("❌ **Status**: Failed\n")
                f.write(f"Error: {real.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        f.write("## Summary\n\n")
        summary = metrics.get("summary", {})
        f.write(f"- Total analyses run: {summary.get('total_analyses', 0)}\n")
        f.write(f"- Successful analyses: {summary.get('successful_analyses', 0)}\n")
        f.write(f"- Methods used: {', '.join(summary.get('causal_discovery_methods', []))}\n\n")
        
        if summary.get("key_findings"):
            f.write("### Key Findings\n\n")
            for finding in summary["key_findings"]:
                f.write(f"- {finding}\n")
    
    print(f"✅ Report saved to: {report_file}")
    return report_file

def main():
    """
    Main function to build all metrics.
    """
    print("\n" + "="*60)
    print("CAUSAL DISCOVERY METRICS BUILD SYSTEM")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # Build metrics
    metrics = build_comprehensive_metrics()
    
    # Generate report
    report_file = generate_markdown_report(metrics)
    
    print("\n" + "="*60)
    print("BUILD COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now()}")
    print(f"\nOutputs:")
    print(f"  - Metrics: Python/latest_metrics.json")
    print(f"  - Report: {report_file.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
