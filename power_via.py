import subprocess
import re
import time
import signal
import os
import argparse

def get_kria_power_stats():
    """
    Calls xmutil to get power stats and returns a dictionary of values.
    """
    try:
        # Execute the xmutil command
        result = subprocess.run(
            ['xmutil', 'xlnx_platformstats', '-p'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout

        # Parse the output to find the power section
        power_stats = {}
        for line in output.splitlines():
            if line.startswith("SOM total power"):
                # Example line: "SOM total power                                         :     5140 mW"
                line_parts = line.split(':')
                if len(line_parts) == 2:
                    value_unit = line_parts[1].strip()
                    match = re.match(r'([\d\.]+)\s*(\w+)', value_unit)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2)
                        power_stats = {'value': value, 'unit': unit}
                        return power_stats
        return power_stats

    except subprocess.CalledProcessError as e:
        print(f"Error executing xmutil: {e}")
        return None
    except FileNotFoundError:
        print("xmutil command not found. Make sure it's in your PATH.")
        return None

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
    parser = argparse.ArgumentParser(description="Monitor and log Kria power stats.")
    parser.add_argument('--power_file', type=str, default="power_log.txt", help="File to log average power.")
    args = parser.parse_args()
    power_file = args.power_file

    print("Collecting power data...")
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    i = 0
    while True:
        stats = get_kria_power_stats()
        if stats:
            power_values.append(stats['value'])
            # print(f"Time: {i+1}s, Power: {stats['value']:.2f} {stats['unit']}")
        time.sleep(1)
        i+=1
