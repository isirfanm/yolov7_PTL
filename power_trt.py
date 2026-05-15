import subprocess
import re
import time
import signal
import os
import argparse
# sudo pip3 install jetson-stats
from jtop import jtop


def get_jetson_power_stats(jetson):
    # Get current power statistics
    power = jetson.power

    return {
        'value': power['tot']['curr'],
        'unit': 'mW',
    }


power_values = []
power_file = "power_log.txt"


def log_average_power():
    if os.path.exists(power_file):
        os.remove(power_file)

    if power_values:
        avg_power = sum(power_values) / len(power_values)
        with open(power_file, "a") as f:
            f.write(f"{avg_power:.2f}\n")
        print(f"Logged average power: {avg_power:.2f}")


def handle_exit(signum, frame):
    log_average_power()
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Monitor and log Kria power stats.")
    parser.add_argument('--power_file', type=str,
                        default="power_log.txt", help="File to log average power.")
    args = parser.parse_args()
    power_file = args.power_file

    print("Collecting power data...")
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    i = 0
    while True:
        # Initialize jtop
        with jtop() as jetson:
            # Check if jtop is running and connected
            if jetson.ok():
                stats = get_jetson_power_stats(jetson)
                if stats:
                    power_values.append(stats['value'])
                    # print(f"Time: {i+1}s, Power: {stats['value']:.2f} {stats['unit']}")
                time.sleep(1)
                i += 1
