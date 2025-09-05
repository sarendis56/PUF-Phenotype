#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from pathlib import Path

def run_experiment(num_devices, batch_size=32, data_path="../data"):
    print(f"\n{'='*70}")
    print(f"Running experiment with {num_devices} devices")
    print(f"{'='*70}")

    # Use the virtual environment's Python directly
    venv_python = Path(".venv/bin/python3")
    if not venv_python.exists():
        venv_python = Path(".venv/Scripts/python.exe")  # Windows path

    if not venv_python.exists():
        print("Error: Virtual environment not found. Please run from the project directory.")
        return False

    cmd = [
        str(venv_python), "main.py",
        "--data_path", data_path,
        "--num_devices", str(num_devices),
        "--batch_size", str(batch_size),
        "--save_path", "results",
        "--no_plots"  # Skip plots for batch processing
    ]

    # Set up environment variables for GPU acceleration
    env = os.environ.copy()
    cuda_lib_path = ".venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    if os.path.exists(cuda_lib_path):
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}"
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment with {num_devices} devices:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nExperiment completed in {duration:.2f} seconds")
    
    return True

def summarize_results():
    """Summarize results from all experiments"""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    results_dir = Path("results")
    
    for num_devices in [3, 4, 5]:
        summary_file = results_dir / f"dpan_summary_{num_devices}_devices.csv"
        
        if summary_file.exists():
            print(f"\n{num_devices} Devices Results:")
            print("-" * 30)
            
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                
            # Print header
            header = lines[0].strip().split(',')
            print(f"{'Classifier':<20} {'Accuracy':<10} {'F1-Score':<10} {'CV Mean':<12}")
            print("-" * 60)
            
            # Print results
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        classifier = parts[0]
                        accuracy = float(parts[1])
                        f1_score = float(parts[2])
                        cv_mean = float(parts[3])
                        
                        print(f"{classifier:<20} {accuracy:<10.4f} {f1_score:<10.4f} {cv_mean:<12.4f}")
        else:
            print(f"\n{num_devices} Devices: Results file not found")

def main():
    # Configuration
    batch_size = 32
    data_path = "../data"
    
    # Check if data path exists
    if not Path(data_path).exists():
        print(f"Error: Data path '{data_path}' does not exist")
        print("Please make sure the dataset is properly extracted")
        sys.exit(1)
    
    # Run experiments for each device configuration
    successful_experiments = 0
    total_experiments = 3
    
    for num_devices in [3, 4, 5]:
        success = run_experiment(num_devices, batch_size, data_path)
        if success:
            successful_experiments += 1
        else:
            print(f"Failed to complete experiment with {num_devices} devices")
    
    # Summarize results
    if successful_experiments > 0:
        summarize_results()
    
    print(f"\n{'='*70}")
    print(f"Batch experiments completed: {successful_experiments}/{total_experiments} successful")
    
    if successful_experiments == total_experiments:
        print("✓ All experiments completed successfully!")
        print(f"\n{'='*70}")
        
    else:
        print("✗ Some experiments failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
