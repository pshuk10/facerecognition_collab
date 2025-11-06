#!/usr/bin/env python3
import sys
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adjust subnet base and worker count as needed
SUBNET_BASE = "10.235.207."
START = 1
END = 254
MAX_WORKERS = 100
PING_TIMEOUT_SECONDS = 6  # overall timeout for each ping process

def make_ping_cmd(ip: str):
    """
    Return a list of command args for ping depending on the platform.
    We rely on subprocess.run(timeout=...) to kill long pings, so we keep the
    ping args minimal and portable.
    """
    system = platform.system().lower()
    if system == "windows":
        # Windows: -n 1 => one echo request
        return ["ping", "-n", "1", ip]
    else:
        # Linux / macOS: -c 1 => one packet
        return ["ping", "-c", "1", ip]

def ping_one(ip: str) -> tuple:
    """
    Ping single IP. Returns (ip, is_up, stdout/stderr or error message).
    Uses subprocess.run with timeout so it won't hang.
    """
    cmd = make_ping_cmd(ip)
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PING_TIMEOUT_SECONDS
        )
        # Returncode 0 usually means success (reachable)
        is_up = (res.returncode == 0)
        return ip, is_up, res.stdout.decode(errors="ignore")
    except subprocess.TimeoutExpired:
        return ip, False, "timeout"
    except Exception as e:
        return ip, False, f"error: {e}"

def main():
    ips = [f"{SUBNET_BASE}{i}" for i in range(START, END + 1)]
    print(f"Scanning {len(ips)} IPs on {SUBNET_BASE}0/24 ...")

    # Use a ThreadPool to run ping_one concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(ping_one, ip): ip for ip in ips}
        for fut in as_completed(futures):
            ip, is_up, info = fut.result()
            if is_up:
                # Print reachable IPs as they are found
                print(f"{ip} is up")
            # Optionally print unreachable info (comment out if noisy)
            # else:
            #     print(f"{ip} is down ({info})")

if __name__ == "__main__":
    main()
