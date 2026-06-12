import subprocess
import sys

result = subprocess.run([sys.executable, "train_all.py"], capture_output=True, text=True, timeout=300)
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn Code: {result.returncode}")
