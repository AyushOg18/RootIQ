"""Find local copies of ai4i2020 dataset or create it from provided data."""
import os
import glob
import shutil

ROOT_DIR = "f:\\GITHUB\\RootIQ"
RAW_TARGET = os.path.join(ROOT_DIR, "data", "raw", "ai4i2020.csv")
os.makedirs(os.path.join(ROOT_DIR, "data", "raw"), exist_ok=True)

candidate_patterns = [
    os.path.join(ROOT_DIR, "*ai4i*.*"),
    os.path.join(ROOT_DIR, "*Dataset*.*"),
    os.path.join(ROOT_DIR, "..", "*ai4i*.*"),
    os.path.join(ROOT_DIR, "..", "*Dataset*.*"),
    "C:\\Users\\hp\\Downloads\\*ai4i*.*",
    "C:\\Users\\hp\\Downloads\\*Dataset*.*",
]

found = []
for pattern in candidate_patterns:
    for f in glob.glob(pattern):
        found.append(f)

print("Found files:", found)

copied = False
for f in found:
    if f.endswith(".csv"):
        print(f"Copying {f} to {RAW_TARGET}")
        shutil.copyfile(f, RAW_TARGET)
        copied = True
        break

if not copied and not os.path.exists(RAW_TARGET):
    print("No CSV file copied yet.")
