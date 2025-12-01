#!/usr/bin/env python3
"""
Training Monitor - Real-time training progress dashboard.

Displays live progress of model training including:
- Process status (running/stopped)
- Recent log output
- File completion status
- Resource usage (CPU, memory)

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --once    # Single check without loop
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def check_process_running(script_name: str) -> tuple:
    """Check if a Python script is currently running."""
    if not HAS_PSUTIL:
        return False, None
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in proc.info['name'].lower():
                if any(script_name in str(arg) for arg in cmdline):
                    return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, None


def get_system_resources() -> tuple:
    """Get current system resource usage."""
    if not HAS_PSUTIL:
        return None, None
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory().percent
    return cpu, memory


def get_file_info(filepath: Path) -> dict:
    """Get file size and modification time."""
    if not filepath.exists():
        return None
    
    stat = filepath.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    age = datetime.now() - mtime
    
    if age.days > 0:
        age_str = f"{age.days}d ago"
    elif age.seconds > 3600:
        age_str = f"{age.seconds // 3600}h ago"
    else:
        age_str = f"{age.seconds // 60}m ago"
    
    return {'size_mb': size_mb, 'mtime': mtime, 'age_str': age_str}


def tail_file(filepath: Path, n: int = 5) -> list:
    """Get last n lines of a file."""
    if not filepath.exists():
        return []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return [line.rstrip()[:100] for line in lines[-n:]]
    except Exception:
        return []


def display_dashboard():
    """Display the training dashboard."""
    clear_screen()
    base_dir = Path(__file__).parent.parent
    
    # Header
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}   TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")
    
    # System resources
    if HAS_PSUTIL:
        cpu, memory = get_system_resources()
        print(f"{Colors.BOLD}System:{Colors.END} CPU {cpu:.1f}% | Memory {memory:.1f}%\n")
    
    # Process status
    print(f"{Colors.BOLD}Process Status:{Colors.END}")
    processes = [
        ('03_gbrt_model.py', 'GBRT Model'),
        ('03_train_new_models.py', 'Elastic Net / Fama-French'),
        ('02_baseline_benchmark.py', 'OLS-3 Baseline'),
    ]
    
    any_running = False
    for script, name in processes:
        running, pid = check_process_running(script)
        if running:
            any_running = True
            print(f"  {Colors.GREEN}▶ {name:30s}{Colors.END} (PID: {pid})")
        else:
            print(f"  {Colors.YELLOW}○ {name:30s}{Colors.END}")
    
    print()
    
    # File completion status
    print(f"{Colors.BOLD}Output Files:{Colors.END}")
    files = [
        ('data/processed/train_data.parquet', 'Training data'),
        ('data/processed/test_data.parquet', 'Test data'),
        ('results/predictions/benchmark_predictions.parquet', 'OLS-3 predictions'),
        ('results/predictions/gbrt_predictions.parquet', 'GBRT predictions'),
        ('results/predictions/elastic_net_predictions.parquet', 'Elastic Net predictions'),
        ('results/predictions/fama_french_predictions.parquet', 'Fama-French predictions'),
    ]
    
    for filepath, description in files:
        info = get_file_info(base_dir / filepath)
        if info:
            print(f"  {Colors.GREEN}✓{Colors.END} {description:35s} ({info['size_mb']:.1f}MB, {info['age_str']})")
        else:
            print(f"  {Colors.RED}○{Colors.END} {description:35s}")
    
    print()
    
    # Check for log files and show recent output
    log_patterns = ['*.log', 'training*.log']
    log_files = list(base_dir.glob('*.log'))
    
    if log_files:
        most_recent = max(log_files, key=lambda f: f.stat().st_mtime)
        print(f"{Colors.BOLD}Recent Log ({most_recent.name}):{Colors.END}")
        lines = tail_file(most_recent, 5)
        for line in lines:
            print(f"  {line}")
        print()
    
    # Footer
    print(f"{Colors.YELLOW}{'─' * 70}{Colors.END}")
    if any_running:
        print(f"Training in progress... Press {Colors.BOLD}Ctrl+C{Colors.END} to exit monitor")
    else:
        print(f"{Colors.GREEN}All processes complete.{Colors.END}")
    print(f"{Colors.YELLOW}{'─' * 70}{Colors.END}")
    
    return any_running


def main():
    """Main monitoring loop."""
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    args = parser.parse_args()
    
    if args.once:
        display_dashboard()
        return
    
    try:
        while True:
            display_dashboard()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{Colors.GREEN}Monitor stopped.{Colors.END}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()

