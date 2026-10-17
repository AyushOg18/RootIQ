"""Copy the user-uploaded dataset directly to data/raw/ai4i2020.csv."""
import os
import shutil

SOURCE_PATH = "C:/Users/hp/.gemini/antigravity/brain/0b1ecad1-c981-4252-8edb-f36db472f9fe/.user_uploaded/media_1792229826352_d3d79b42.csv"
DEST_PATH = "f:/GITHUB/RootIQ/data/raw/ai4i2020.csv"

os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)
shutil.copyfile(SOURCE_PATH, DEST_PATH)

print(f"Successfully copied user dataset ({os.path.getsize(DEST_PATH)} bytes) to {DEST_PATH}")
