import psutil
import re
import time
import signal
import os
import argparse


cpu_values = []
cpu_file = "cpu_log.txt"

def log_average_cpu():
    if os.path.exists(cpu_file):
        os.remove(cpu_file)

    if cpu_values:
        avg = sum(cpu_values) / len(cpu_values)
        with open(cpu_file, "a") as f:
            f.write(f"{avg:.2f}\n")
        print(f"Logged average cpu: {avg:.2f}")

def handle_exit(signum, frame):
    log_average_cpu()
    exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monitor and log Kria cpu stats.")
    parser.add_argument('--cpu_file', type=str, default="cpu_log.txt", help="File to log average cpu.")
    args = parser.parse_args()
    power_file = args.power_file

    print("Collecting cpu data...")
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    i = 0
    while True:
        stats = psutil.cpu_percent(interval=1)
        cpu_values.append(stats)
        time.sleep(1)
        i+=1
