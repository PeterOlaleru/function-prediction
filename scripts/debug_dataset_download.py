import os
import sys
from pathlib import Path
import subprocess
import zipfile
import shutil

# 1. Load Secrets from .env (Local Debugging)
# Assuming script is in scripts/ folder, so .env is in parent
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    print(f"Loading secrets from {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v
else:
    print("WARNING: .env not found. Ensure KAGGLE_USERNAME/KEY are set.")

# 2. Define Local Work Root
# We will download to a specific debug folder to avoid messing up the main workspace
# Using absolute path based on script location to be safe
PROJECT_ROOT = Path(__file__).parent.parent
DEBUG_ROOT = PROJECT_ROOT / 'artefacts_local' / 'debug_download'
DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Debug Root: {DEBUG_ROOT}")

# 3. Check Kaggle Auth
try:
    subprocess.run(['kaggle', '--version'], check=True, capture_output=True)
    print("Kaggle CLI detected.")
except Exception as e:
    print("Kaggle CLI not found or not working. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])

print(f"User: {os.environ.get('KAGGLE_USERNAME')}")
dataset_id = os.environ.get('CAFA_CHECKPOINT_DATASET_ID')
print(f"Dataset ID: {dataset_id}")

if not dataset_id:
    print("ERROR: CAFA_CHECKPOINT_DATASET_ID is not set. Cannot proceed.")
    sys.exit(1)

# CELL 2: Robust Download & Unzip Function
def list_dataset_files():
    print(f"\n--- Listing files in {dataset_id} ---")
    try:
        subprocess.run(['kaggle', 'datasets', 'files', dataset_id], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED to list files: {e}")

def download_file(file_path, target_dir):
    print(f"\n--- Processing {file_path} ---")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download
    print(f"Downloading {file_path} from {dataset_id}...")
    try:
        # Note: -f takes the remote file path. 
        # If the file is in a folder 'features/file.npy', we pass 'features/file.npy'
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_id, '-f', file_path, '-p', str(target_dir)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"FAILED to download {file_path}: {e}")
        return

    # Kaggle downloads individual files as zips if they are large, or just the file?
    # Usually it downloads as file_name.zip if it's a single file download.
    # Let's check what we got.
    
    downloaded_name = Path(file_path).name
    possible_zip = target_dir / (downloaded_name + '.zip')
    possible_file = target_dir / downloaded_name
    
    if possible_zip.exists():
        print(f"  Downloaded as zip: {possible_zip}")
        print(f"  Unzipping...")
        with zipfile.ZipFile(possible_zip, 'r') as zf:
            zf.extractall(target_dir)
        possible_zip.unlink()
        print("  Unzipped and cleaned up.")
    elif possible_file.exists():
        print(f"  Downloaded as raw file: {possible_file}")
    else:
        print(f"ERROR: Could not find downloaded file for {file_path}")

# List files first
list_dataset_files()

# Define the full list of files we expect to need
files_to_download = [
    # Embeddings
    'features/train_embeds_t5.npy',
    'features/train_embeds_esm2.npy',
    'features/test_embeds_t5.npy',
    'features/test_embeds_esm2.npy',
    
    # Text Features (Corrected Names)
    'features/text_vectorizer.joblib',      # Was tfidf_vectorizer.joblib
    'features/train_embeds_text.npy',       # Was train_text_features.npy
    'features/test_embeds_text.npy',        # Was test_text_features.npy
    
    # External / Props
    'external/prop_train_no_kaggle.tsv',
    'external/prop_test_no_kaggle.tsv',
    'external/entryid_text.tsv',

    # Parsed Metadata (Required for Training)
    'parsed/train_terms.parquet',
    'parsed/train_seq.feather',
    'parsed/train_taxa.feather'
]

print(f"\n--- Starting Batch Download of {len(files_to_download)} files ---")

for fname in files_to_download:
    # Determine target directory based on file structure
    if 'features/' in fname:
        target = DEBUG_ROOT / 'features'
    elif 'external/' in fname:
        target = DEBUG_ROOT / 'external'
    else:
        target = DEBUG_ROOT
        
    download_file(fname, target)

print("\n--- Download Check Complete ---")

# CELL 3: File System Audit
print("\n=== File System Audit ===")
for root, dirs, files in os.walk(DEBUG_ROOT):
    level = root.replace(str(DEBUG_ROOT), '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        size_mb = (Path(root) / f).stat().st_size / (1024**2)
        print(f"{subindent}{f} ({size_mb:.2f} MB)")

print("\n=== Verification Check ===")
# Mimic the check in the main notebook
required = [
    'features/train_embeds_t5.npy',
    'external/prop_train_no_kaggle.tsv', # Check uncompressed
    'external/prop_train_no_kaggle.tsv.gz' # Check compressed
]

for r in required:
    p = DEBUG_ROOT / r
    exists = p.exists()
    print(f"Checking {r}: {exists}")
    if not exists:
        # Try resolving .gz
        if p.suffix == '.tsv' and p.with_suffix('.tsv.gz').exists():
             print(f"  -> Found as .tsv.gz instead")
        elif p.suffix == '.gz' and p.with_suffix('').exists():
             print(f"  -> Found as uncompressed instead")
