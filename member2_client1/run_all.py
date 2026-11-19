"""HW4 - Member 2: run the whole Client 1 pipeline in one go (about 1-2 minutes on a laptop CPU).

    python run_all.py              # full run: 20 FL rounds
    python run_all.py --quick      # smoke test: 2 FL rounds, just to check everything works

Steps:
  1. local_train.py --all  local-only test of Client 1 + baselines (clients 2/3, centralised)
  2. simulate_fl.py        Flower server (FedAvg) + 3 clients on this machine
  3. plot_results.py       figures for the report
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    """Run one step with the same Python interpreter; stop if it fails."""
    print("\n=== python %s ===" % " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=str(HERE), check=True)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run("local_train.py", "--all")
    run("simulate_fl.py", *(["--rounds", "2"] if quick else []))
    run("plot_results.py")
    print("\nDone. Results and figures are in", HERE.parent / "results" / "member2_client1")
