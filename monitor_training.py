#!/usr/bin/env python3
"""
Real-time Training Monitor

Displays live progress of model training including:
- Current step and estimated time remaining
- Resource usage (CPU, memory)
- Log file tails
- Visual progress bars

Usage: python monitor_training.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import psutil

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_file_size(filepath):
    """Get file size in human-readable format."""
    if not os.path.exists(filepath):
        return "N/A"
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def get_file_age(filepath):
    """Get time since file was last modified."""
    if not os.path.exists(filepath):
        return "N/A"
    mtime = os.path.getmtime(filepath)
    age = time.time() - mtime
    if age < 60:
        return f"{int(age)}s ago"
    elif age < 3600:
        return f"{int(age/60)}m ago"
    else:
        return f"{int(age/3600)}h ago"

def tail_file(filepath, n=5):
    """Get last n lines of a file."""
    if not os.path.exists(filepath):
        return ["File not found"]
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return [line.rstrip() for line in lines[-n:]]
    except Exception as e:
        return [f"Error reading file: {e}"]

def check_process_running(script_name):
    """Check if a Python script is currently running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in proc.info['name'].lower():
                if any(script_name in arg for arg in cmdline):
                    return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, None

def get_system_resources():
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return cpu_percent, memory.percent

def progress_bar(current, total, width=40):
    """Create a text progress bar."""
    if total == 0:
        return "[" + "?" * width + "]"
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percent = 100 * current / total
    return f"[{bar}] {percent:.1f}%"

def check_training_status():
    """Check status of all training scripts and data files."""
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    
    status = {
        'data_prep': {
            'script': 'src/01_data_preparation.py',
            'output': 'data/train_data.parquet',
            'test': 'data/test_data.parquet',
            'estimated_time': '2 min',
        },
        'baseline': {
            'script': 'src/02_baseline_benchmark.py',
            'output': 'results/predictions/benchmark_predictions.parquet',
            'estimated_time': '2 min',
        },
        'gbrt': {
            'script': 'src/03_gbrt_model.py',
            'output': 'results/predictions/gbrt_predictions.parquet',
            'estimated_time': '5 hours',
        },
        'new_models': {
            'script': 'src/03_train_new_models.py',
            'output': 'results/predictions/elastic_net_predictions.parquet',
            'estimated_time': '20 min',
        },
        'evaluation': {
            'script': 'src/04_evaluation.py',
            'output': 'results/tables/performance_comparison.csv',
            'estimated_time': '5 min',
        },
        'unified': {
            'script': 'src/06_unified_evaluation.py',
            'output': 'results/tables/all_models_performance.csv',
            'estimated_time': '5 min',
        }
    }
    
    for key, info in status.items():
        script_path = base_dir / info['script']
        output_path = base_dir / info['output']
        
        # Check if script is running
        running, pid = check_process_running(info['script'])
        info['running'] = running
        info['pid'] = pid
        
        # Check if output exists
        info['output_exists'] = output_path.exists()
        if info['output_exists']:
            info['output_size'] = get_file_size(output_path)
            info['output_age'] = get_file_age(output_path)
        else:
            info['output_size'] = "N/A"
            info['output_age'] = "N/A"
    
    return status

def display_dashboard():
    """Display the training dashboard."""
    clear_screen()
    
    # Header
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}   EMPIRICAL ASSET PRICING - TRAINING MONITOR{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    # Current time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BOLD}Current Time:{Colors.END} {now}\n")
    
    # System resources
    cpu, memory = get_system_resources()
    print(f"{Colors.BOLD}System Resources:{Colors.END}")
    print(f"  CPU Usage:    {progress_bar(cpu, 100, 30)} {cpu:.1f}%")
    print(f"  Memory Usage: {progress_bar(memory, 100, 30)} {memory:.1f}%\n")
    
    # Training status
    status = check_training_status()
    
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.END}")
    print(f"{Colors.BOLD}TRAINING PIPELINE STATUS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.END}\n")
    
    steps = [
        ('data_prep', '1. Data Preparation'),
        ('baseline', '2. OLS-3 Baseline'),
        ('gbrt', '3. GBRT Model'),
        ('new_models', '4. Elastic Net & Fama-French'),
        ('evaluation', '5. Model Evaluation'),
        ('unified', '6. Unified Comparison'),
    ]
    
    for key, name in steps:
        info = status[key]
        
        # Status indicator
        if info['running']:
            status_icon = f"{Colors.GREEN}▶ RUNNING{Colors.END}"
            status_detail = f"PID: {info['pid']}"
        elif info['output_exists']:
            status_icon = f"{Colors.BLUE}✓ COMPLETE{Colors.END}"
            status_detail = f"({info['output_age']})"
        else:
            status_icon = f"{Colors.YELLOW}○ PENDING{Colors.END}"
            status_detail = f"Est. {info['estimated_time']}"
        
        print(f"{Colors.BOLD}{name:40s}{Colors.END} {status_icon:20s} {status_detail}")
        
        if info['output_exists']:
            print(f"  └─ Output: {info['output']} ({info['output_size']})")
        
        print()
    
    # Log tails
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.END}")
    print(f"{Colors.BOLD}RECENT LOGS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.END}\n")
    
    log_files = [
        ('training.log', 'Training Log'),
        ('src/03_gbrt_model.py.log', 'GBRT Progress'),
    ]
    
    for log_file, label in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            print(f"{Colors.BOLD}{label}:{Colors.END} (last 3 lines)")
            lines = tail_file(log_path, 3)
            for line in lines:
                print(f"  {line[:100]}")  # Truncate long lines
            print()
    
    # Instructions
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.END}")
    print(f"{Colors.BOLD}Press Ctrl+C to exit monitoring{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*80}{Colors.END}")

def main():
    """Main monitoring loop."""
    print(f"{Colors.BOLD}{Colors.GREEN}Starting Training Monitor...{Colors.END}\n")
    time.sleep(1)
    
    try:
        while True:
            display_dashboard()
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{Colors.BOLD}{Colors.GREEN}Monitor stopped.{Colors.END}\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
