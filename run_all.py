"""One command to reproduce everything:  python run_all.py   (add --quick for a fast test)"""
import subprocess
import sys

quick = ["--quick"] if "--quick" in sys.argv else []
subprocess.run([sys.executable, "src/benchmark.py"] + quick, check=True)
subprocess.run([sys.executable, "src/plots.py"], check=True)
print("\nDone. See the results/ folder for results.csv and fig*.png")
