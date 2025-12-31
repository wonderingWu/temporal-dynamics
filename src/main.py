#!/usr/bin/env python3
"""
Main entry point for temporal-dynamics project.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Information-theoretic analysis of temporal dynamics in complex systems"
    )
    
    parser.add_argument(
        "--system",
        choices=["esn", "ising", "logistic", "sandpile", "all"],
        default="all",
        help="System to analyze (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.verbose:
        print(f"Starting temporal-dynamics analysis...")
        print(f"System: {args.system}")
        print(f"Output directory: {output_dir}")
    
    try:
        if args.system == "esn":
            from ESN_criticality_te_fixed import main as run_esn
            run_esn()
        elif args.system == "ising":
            from ising_ais_nmi_quick_revised import main as run_ising
            run_ising()
        elif args.system == "logistic":
            from logicMapTest_fixed import main as run_logistic
            run_logistic()
        elif args.system == "sandpile":
            from sandpile_critical_time_analysis import main as run_sandpile
            run_sandpile()
        elif args.system == "all":
            print("Running all analyses...")
            
            systems = [
                ("esn", "ESN_criticality_te_fixed", "main"),
                ("ising", "ising_ais_nmi_quick_revised", "main"),
                ("logistic", "logicMapTest_fixed", "main"),
                ("sandpile", "sandpile_critical_time_analysis", "main")
            ]
            
            for system_name, module_name, func_name in systems:
                print(f"Running {system_name} analysis...")
                try:
                    module = __import__(module_name)
                    if hasattr(module, func_name):
                        getattr(module, func_name)()
                    else:
                        print(f"Warning: {module_name} has no {func_name} function")
                except Exception as e:
                    print(f"Error running {system_name} analysis: {e}")
                    continue
            
            print("All analyses completed.")
        
        if args.verbose:
            print("Analysis completed successfully!")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()