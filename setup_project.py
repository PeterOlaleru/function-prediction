# Project Scaffolding Script
# Run this to create the folder structure

import os

def create_structure():
    """Create project folder structure"""
    
    folders = [
        'notebooks',
        'src',
        'src/data',
        'src/models',
        'src/evaluation',
        'src/utils',
        'experiments',
        'submissions',
        'logs',
    ]
    
    print("Creating project structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}/")
    
    # Create __init__.py files for Python packages
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py',
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Python package\n')
        print(f"✓ Created: {init_file}")
    
    # Create starter files
    starter_files = {
        'src/data/loaders.py': '''"""Data loading utilities"""

from Bio import SeqIO
import pandas as pd
import numpy as np


class SequenceLoader:
    """Load FASTA sequences"""
    
    def __init__(self, fasta_path):
        self.fasta_path = fasta_path
        self.sequences = {}
        
    def load(self):
        """Load all sequences into memory"""
        for record in SeqIO.parse(self.fasta_path, 'fasta'):
            self.sequences[record.id] = str(record.seq)
        print(f"Loaded {len(self.sequences)} sequences")
        return self.sequences
    
    def get_sequence(self, protein_id):
        """Get single sequence"""
        return self.sequences.get(protein_id, None)


class LabelLoader:
    """Load GO term labels"""
    
    def __init__(self, labels_path):
        self.labels_path = labels_path
        self.labels_df = None
        
    def load(self):
        """Load labels from TSV"""
        self.labels_df = pd.read_csv(self.labels_path, sep='\\t')
        print(f"Loaded {len(self.labels_df)} annotations")
        return self.labels_df
    
    def get_labels_for_protein(self, protein_id):
        """Get all GO terms for a protein"""
        protein_labels = self.labels_df[self.labels_df['EntryID'] == protein_id]
        return protein_labels['term'].tolist()
    
    def get_proteins_for_term(self, go_term):
        """Get all proteins with a GO term"""
        term_proteins = self.labels_df[self.labels_df['term'] == go_term]
        return term_proteins['EntryID'].tolist()
''',
        
        'src/data/features.py': '''"""Feature engineering utilities"""

import numpy as np
from collections import Counter


def calculate_aa_composition(sequence):
    """Calculate amino acid composition (20 features)"""
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_count = Counter(sequence)
    total = len(sequence)
    
    composition = {}
    for aa in aa_list:
        composition[aa] = aa_count.get(aa, 0) / total
    
    return composition


def extract_kmers(sequence, k=3):
    """Extract k-mer frequencies"""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    
    kmer_counts = Counter(kmers)
    total = len(kmers)
    
    # Normalize
    kmer_freq = {kmer: count/total for kmer, count in kmer_counts.items()}
    return kmer_freq


def calculate_sequence_features(sequence):
    """Calculate comprehensive sequence features"""
    features = {}
    
    # Length
    features['length'] = len(sequence)
    
    # AA composition
    aa_comp = calculate_aa_composition(sequence)
    features.update(aa_comp)
    
    # Basic properties
    features['charge'] = sum(1 for aa in sequence if aa in 'KR') - sum(1 for aa in sequence if aa in 'DE')
    features['hydrophobicity'] = sum(1 for aa in sequence if aa in 'AILMFWYV') / len(sequence)
    
    return features
''',
        
        'src/models/baseline.py': '''"""Baseline models"""

import numpy as np
from collections import Counter


class FrequencyBaseline:
    """Predict most frequent GO terms"""
    
    def __init__(self):
        self.term_frequencies = {}
        
    def fit(self, labels_df):
        """Learn term frequencies from training data"""
        term_counts = labels_df['term'].value_counts()
        total = len(labels_df['EntryID'].unique())
        
        self.term_frequencies = {
            term: count / total 
            for term, count in term_counts.items()
        }
        
        print(f"Learned frequencies for {len(self.term_frequencies)} terms")
    
    def predict(self, protein_ids, top_k=100):
        """Predict top-k most frequent terms for all proteins"""
        predictions = {}
        
        # Get top-k terms
        top_terms = sorted(self.term_frequencies.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:top_k]
        
        for protein_id in protein_ids:
            predictions[protein_id] = top_terms
        
        return predictions
''',
        
        'src/evaluation/metrics.py': '''"""Evaluation metrics"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate precision, recall, F1"""
    
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_weighted_f1(y_true, y_pred, ia_weights, threshold=0.5):
    """Calculate IA-weighted F1 score"""
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate TP, FP, FN per term
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum(axis=0)
    
    # Precision and recall per term
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Weight by IA
    weighted_precision = (precision * ia_weights).sum() / ia_weights.sum()
    weighted_recall = (recall * ia_weights).sum() / ia_weights.sum()
    
    # F1
    if weighted_precision + weighted_recall == 0:
        return 0.0
    
    f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    
    return f1
''',
        
        'src/utils/helpers.py': '''"""Helper utilities"""

import os
import json
from datetime import datetime


def create_experiment_folder(experiment_name):
    """Create folder for experiment"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{experiment_name}_{timestamp}"
    folder_path = os.path.join('experiments', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_config(config, folder_path):
    """Save experiment configuration"""
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


def load_config(folder_path):
    """Load experiment configuration"""
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
''',
        
        '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data
data/
*.fasta
*.tsv
*.obo

# Models
experiments/
*.pth
*.h5
*.pkl

# Submissions
submissions/*.tsv
submissions/*.zip

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db
''',
        
        'requirements.txt': '''# Core
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0

# ML
scikit-learn>=1.1.0
torch>=2.0.0
transformers>=4.30.0

# Bio
biopython>=1.80
obonet>=0.3.0
networkx>=2.8

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Notebook
jupyter>=1.0.0
ipykernel>=6.15.0

# Utils
tqdm>=4.64.0
''',
        
        'README.md': '''# CAFA-6 Protein Function Prediction

## Project Overview
Predicting protein functions from amino acid sequences using machine learning.

## Getting Started

1. **Read the documentation:**
   - `EXPLAINER.md` - Simple explanation
   - `PLAN.md` - Detailed roadmap
   - `QUICK_START.md` - Quick reference

2. **Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. **Run EDA:**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## Project Structure
```
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
│   ├── data/          # Data loading
│   ├── models/        # Model definitions
│   ├── evaluation/    # Metrics
│   └── utils/         # Helpers
├── experiments/       # Model experiments
└── submissions/       # Final predictions
```

## Quick Links
- [Competition Overview](docs/overview.md)
- [Data Description](docs/dataset_description.md)

## Progress Tracker
- [ ] Setup & EDA
- [ ] Data processing
- [ ] Baseline models
- [ ] Deep learning
- [ ] Submission

## Contact
Your Name - your.email@example.com
'''
    }
    
    for file_path, content in starter_files.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✓ Created: {file_path}")
    
    print("\\n✅ Project structure created successfully!")
    print("\\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Open: QUICK_START.md")
    print("3. Start: jupyter notebook")


if __name__ == '__main__':
    create_structure()
