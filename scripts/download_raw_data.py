"""Standard-library-only downloader for UCI AI4I 2020 dataset."""
import io
import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DATA_FILE = os.path.join(DATA_DIR, "ai4i2020.csv")
UCI_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 1000:
        print(f"Dataset already exists at {DATA_FILE} ({os.path.getsize(DATA_FILE)} bytes)")
        return
    print(f"Downloading from {UCI_URL}...")
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for fname in z.namelist():
            if fname.endswith(".csv"):
                data = z.read(fname)
                with open(DATA_FILE, "wb") as f:
                    f.write(data)
                print(f"Extracted {fname} to {DATA_FILE} ({len(data)} bytes)")
                return
    raise RuntimeError("No CSV found in zip")

if __name__ == "__main__":
    download()
