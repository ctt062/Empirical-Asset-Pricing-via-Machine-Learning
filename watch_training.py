#!/usr/bin/env python3
"""
Simple Real-Time Training Monitor

Monitors the new_models_training.log file and displays:
- Current month being processed
- Progress percentage
- Estimated time remaining
- Recent log messages

Usage: python3 watch_training.py
"""

import os
import sys
import time
import re
from datetime import datetime, timedelta

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def parse_log_for_progress(log_file):
    """Parse log file to extract training progress."""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for progress indicators
        current_month = 0
        total_months = 252
        model_name = "Unknown"
        recent_lines = []
        start_time = None
        
        for line in lines[-100:]:  # Check last 100 lines
            # Look for "[X/252]" pattern
            match = re.search(r'\[(\d+)/(\d+)\]', line)
            if match:
                current_month = int(match.group(1))
                total_months = int(match.group(2))
            
            # Detect which model
            if 'ELASTIC NET' in line:
                model_name = "Elastic Net"
            elif 'FAMA-FRENCH' in line:
                model_name = "Fama-French"
            
            # Get start time
            if 'Start time:' in line:
                try:
                    time_str = line.split('Start time:')[1].strip()
                    start_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass
        
        # Get last 5 lines
        recent_lines = [line.strip() for line in lines[-5:] if line.strip()]
        
        return {
            'current_month': current_month,
            'total_months': total_months,
            'model_name': model_name,
            'recent_lines': recent_lines,
            'start_time': start_time,
            'log_size': os.path.getsize(log_file)
        }
    except Exception as e:
        return {'error': str(e)}

def format_time(seconds):
    """Format seconds into readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def display_progress(log_file='new_models_training.log'):
    """Display training progress."""
    clear_screen()
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          TRAINING MONITOR - Real-Time Progress                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Parse log
    info = parse_log_for_progress(log_file)
    
    if info is None:
        print("⚠️  Log file not found: new_models_training.log")
        print()
        print("Make sure training is running:")
        print("  $ python3 src/03_train_new_models.py 2>&1 | tee new_models_training.log")
        return
    
    if 'error' in info:
        print(f"❌ Error reading log: {info['error']}")
        return
    
    # Display current status
    current = info['current_month']
    total = info['total_months']
    model = info['model_name']
    
    print(f"📊 Current Model: {model}")
    print(f"🔢 Progress: Month {current}/{total}")
    
    # Progress bar
    if total > 0:
        percent = (current / total) * 100
        bar_length = 50
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"📈 [{bar}] {percent:.1f}%")
    
    # Time estimates
    if info['start_time'] and current > 0:
        elapsed = (datetime.now() - info['start_time']).total_seconds()
        time_per_month = elapsed / current
        remaining_months = total - current
        estimated_remaining = time_per_month * remaining_months
        
        print(f"⏱️  Elapsed: {format_time(elapsed)}")
        print(f"⏰ Remaining: ~{format_time(estimated_remaining)}")
        
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        print(f"🎯 ETA: {eta.strftime('%H:%M:%S')}")
    
    print()
    print("─" * 70)
    print("📝 Recent Log Messages:")
    print("─" * 70)
    
    for line in info['recent_lines']:
        # Truncate very long lines
        if len(line) > 100:
            line = line[:97] + "..."
        print(line)
    
    print()
    print("─" * 70)
    print(f"Log size: {info['log_size']/1024/1024:.1f} MB")
    print("Press Ctrl+C to exit • Updating every 5 seconds")
    print("─" * 70)

def main():
    """Main monitoring loop."""
    log_file = 'new_models_training.log'
    
    print("\033[92mStarting training monitor...\033[0m\n")
    time.sleep(1)
    
    try:
        while True:
            display_progress(log_file)
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        clear_screen()
        print("\n\033[92m✓ Monitor stopped.\033[0m\n")
        
        # Show final status
        info = parse_log_for_progress(log_file)
        if info and 'current_month' in info:
            print(f"Final progress: {info['current_month']}/{info['total_months']} months")
        
        sys.exit(0)

if __name__ == "__main__":
    main()
