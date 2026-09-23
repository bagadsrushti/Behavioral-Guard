import argparse
import subprocess
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Start Behavioral Guard real data collection session.")
    parser.add_argument("--session_id", required=True, help="Unique identifier for this session")
    parser.add_argument("--label", required=True, choices=["genuine", "imposter"], help="Label for this session")
    args = parser.parse_args()

    raw_dir = "data/real/raw"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs("data/real/processed", exist_ok=True)

    mouse_out = os.path.join(raw_dir, f"mouse_{args.session_id}.csv")
    ks_out = os.path.join(raw_dir, f"keystroke_{args.session_id}.csv")

    print(f"Starting session '{args.session_id}' [{args.label}]")
    print(f"Mouse log: {mouse_out}")
    print(f"Keystroke log: {ks_out}")
    print("Press Ctrl+C to stop logging...")

    # Start both loggers as subprocesses
    mouse_proc = subprocess.Popen([
        "python", "-m", "src.logging.mouse_logger",
        "--session_id", args.session_id,
        "--label", args.label,
        "--output", mouse_out
    ])

    ks_proc = subprocess.Popen([
        "python", "-m", "src.logging.keystroke_logger",
        "--session_id", args.session_id,
        "--label", args.label,
        "--output", ks_out
    ])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping loggers...")
        mouse_proc.terminate()
        ks_proc.terminate()
        mouse_proc.wait()
        ks_proc.wait()
        print("Session saved successfully.")

if __name__ == "__main__":
    main()
