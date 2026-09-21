#!/usr/bin/env python3
"""
Master Analysis Runner

Execute all statistical tests comparing real NFL outcomes with
matched Bernoulli simulations.

Generates:
1. Monte Carlo simulations (10,000 synthetic NFL histories)
2. Calibration analysis
3. Upsets analysis
4. Team time series analysis (streaks, autocorrelation)
5. Residuals analysis
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_path, description):
    """
    Run a Python script and handle errors.

    Args:
        script_path: Path to script
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} complete\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}\n")
        return False


def main():
    """Run all analyses in sequence."""
    start_time = datetime.now()

    print("\n" + "#" * 80)
    print("# NFL BERNOULLI INDEPENDENCE TEST - FULL ANALYSIS PIPELINE")
    print("#" * 80)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Ensure output directory exists
    output_dir = Path('output/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis pipeline
    analyses = [
        ('scripts/03_bernoulli_simulator.py', 'Monte Carlo Simulation (10,000 histories)'),
        ('scripts/04_analysis_calibration.py', 'Calibration Analysis'),
        ('scripts/05_analysis_upsets.py', 'Upsets Analysis'),
        ('scripts/06_analysis_team_timeseries.py', 'Team Time Series Analysis'),
        ('scripts/07_analysis_residuals.py', 'Residuals Analysis'),
    ]

    results = []
    for script_path, description in analyses:
        success = run_script(script_path, description)
        results.append((description, success))

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "#" * 80)
    print("# ANALYSIS PIPELINE COMPLETE")
    print("#" * 80)

    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")

    print("\nResults Summary:")
    print("-" * 50)
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status:12} | {description}")

    # Check if all succeeded
    all_success = all(success for _, success in results)

    if all_success:
        print("\n✓ All analyses completed successfully!")
        print(f"\nOutputs saved to: {output_dir.absolute()}")

        print("\nGenerated Files:")
        if output_dir.exists():
            for file_path in sorted(output_dir.iterdir()):
                print(f"  - {file_path.name}")

        return 0
    else:
        print("\n✗ Some analyses failed. Check output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
