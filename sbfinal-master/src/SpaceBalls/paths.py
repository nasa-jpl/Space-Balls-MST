# src/awesome_package/paths.py
from pathlib import Path

# Only define this once
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input_files"
CONFIG_DIR = PROJECT_ROOT / "config"
MEDIA_DIR = '/media/monte_share'
OUTPUT_DIR = '/media/monte_share/output_files'