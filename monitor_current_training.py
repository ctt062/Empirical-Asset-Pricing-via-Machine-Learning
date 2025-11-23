#!/usr/bin/env python3
"""
Monitor current training progress in real-time.
Shows GBRT and Elastic Net/Fama-French training progress.
"""

import subprocess
import time
import os
from datetime import datetime

def check_processes():
    """Check if training processes are running."""
    gbrt = subprocess.run(['pgrep', '-f', 'src/03_gbrt_model.py'], capture_output=True).returncode == 0
    new_models = subprocess.run(['pgrep', '-f', 'src/03_train_new_models.py'], capture_output=True).returncode == 0
    return gbrt, new_models

def get_last_lines(filename, n=5):
    """Get last n lines from a file."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-n:])
    except FileNotFoundError:
        return f"Log file not found: {filename}"
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"

def main():
    """Monitor training progress."""
    print("Starting training monitor...")
    print("Press Ctrl+C to exit")
    print()
    
    try:
        while True:
            os.system('clear')
            
            # Header
            print("=" * 80)
            print("🚀 TRAINING PROGRESS MONITOR")
            print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()
            
            # Check status
            gbrt_running, new_models_running = check_processes()
            
            # Status
            print("📊 Process Status:")
            print(f"  GBRT:       {'🟢 RUNNING' if gbrt_running else '🔴 STOPPED'}")
            print(f"  New Models: {'🟢 RUNNING' if new_models_running else '🔴 STOPPED'}")
            print()
            
            # GBRT Progress
            print("=" * 80)
            print("GBRT Training Progress (gbrt_final.log):")
            print("=" * 80)
            print(get_last_lines('gbrt_final.log', 10))
            print()
            
            # New Models Progress
            print("=" * 80)
            print("Elastic Net & Fama-French Progress (new_models_final.log):")
            print("=" * 80)
            print(get_last_lines('new_models_final.log', 10))
            print()
            
            # Check if done
            if not gbrt_running and not new_models_running:
                print("=" * 80)
                print("✅ ALL TRAINING COMPLETED!")
                print("=" * 80)
                print()
                print("Next step: Run evaluation")
                print("  python3 src/06_unified_evaluation.py")
                break
            
            # Refresh interval
            time.sleep(5)  # Update every 5 seconds for more frequent updates
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print()
        if gbrt_running or new_models_running:
            print("⚠️  Training is still running in the background!")
            print("Check status: ps aux | grep 'python3 src/03'")

if __name__ == '__main__':
    main()
