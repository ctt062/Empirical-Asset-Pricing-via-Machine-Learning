#!/usr/bin/env python3
"""
Monitor retraining progress for both GBRT and new models.
"""
import time
import os
from datetime import datetime

def monitor():
    """Monitor training progress."""
    print("="*80)
    print("RETRAINING MONITOR - Press Ctrl+C to stop")
    print("="*80)
    print()
    
    while True:
        os.system('clear')
        print("="*80)
        print(f"⚠️  RETRAINING WITH LESS PREDICTABLE SYNTHETIC DATA")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print()
        
        # Check processes
        gbrt_running = os.system('pgrep -f "python3 src/03_gbrt_model.py" > /dev/null 2>&1') == 0
        new_models_running = os.system('pgrep -f "python3 src/03_train_new_models.py" > /dev/null 2>&1') == 0
        
        print("📊 Process Status:")
        print(f"  GBRT:       {'🟢 RUNNING' if gbrt_running else '🔴 STOPPED'}")
        print(f"  New Models: {'🟢 RUNNING' if new_models_running else '🔴 STOPPED'}")
        print()
        
        # GBRT progress
        print("="*80)
        print("GBRT Progress:")
        print("="*80)
        if os.path.exists('gbrt_retrain.log'):
            os.system('tail -5 gbrt_retrain.log')
        else:
            print("  Log file not found")
        print()
        
        # New models progress
        print("="*80)
        print("Elastic Net & Fama-French Progress:")
        print("="*80)
        if os.path.exists('new_models_retrain.log'):
            os.system('tail -5 new_models_retrain.log')
        else:
            print("  Log file not found")
        print()
        
        # Check completion
        if not gbrt_running and not new_models_running:
            print("="*80)
            print("✅ ALL TRAINING COMPLETED!")
            print("="*80)
            print("\nNext step: Run evaluation")
            print("  python3 src/06_unified_evaluation.py")
            break
        
        print("="*80)
        print("Refreshing in 30 seconds... (Ctrl+C to stop)")
        print("="*80)
        
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
