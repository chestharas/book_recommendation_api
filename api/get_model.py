import subprocess
from pathlib import Path
import sys

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCRAPE_DIR = BASE_DIR / 'scrape'
MODEL_DIR = BASE_DIR / 'models'

# Define script paths
SCRAPE_LINK_SCRIPT = SCRAPE_DIR / 'scrape_link.py'
SCRAPE_BOOK_SCRIPT = SCRAPE_DIR / 'scrape_book.py'
TRAIN_MODEL_SCRIPT = MODEL_DIR / 'train_model.py'

def run_script_live(script_path):
    print(f"\n🚀 Running: {script_path}")
    
    process = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],  # "-u" forces unbuffered output
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        print(f"❌ Script {script_path} exited with errors (code {process.returncode})")

if __name__ == "__main__":
    print("📦 Starting data and model pipeline...")

    run_script_live(SCRAPE_LINK_SCRIPT)
    run_script_live(SCRAPE_BOOK_SCRIPT)
    run_script_live(TRAIN_MODEL_SCRIPT)

    print("\n✅ Pipeline completed successfully.")
