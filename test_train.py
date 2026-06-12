import os
import sys

print("DEBUG: Starting test script", flush=True)

try:
    import pandas as pd
    print("DEBUG: pandas imported", flush=True)
except Exception as e:
    print(f"ERROR importing pandas: {e}", flush=True)
    sys.exit(1)

try:
    from config import DISEASE_CONFIG
    print("DEBUG: config imported", flush=True)
except Exception as e:
    print(f"ERROR importing config: {e}", flush=True)
    sys.exit(1)

try:
    os.makedirs("models", exist_ok=True)
    print("DEBUG: models directory created", flush=True)
except Exception as e:
    print(f"ERROR creating directory: {e}", flush=True)
    sys.exit(1)

try:
    cfg = DISEASE_CONFIG['Diabetes']
    df = pd.read_csv(cfg['file'])
    print(f"DEBUG: Loaded Diabetes data, shape: {df.shape}", flush=True)
except Exception as e:
    print(f"ERROR loading data: {e}", flush=True)
    sys.exit(1)

print("DEBUG: Test passed", flush=True)
