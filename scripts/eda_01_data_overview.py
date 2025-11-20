import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import networkx as nx
import obonet
from statistics import mean

# Set non-interactive backend for saving plots
plt.switch_backend('Agg')

sns.set_context("talk")
sns.set_style("whitegrid")

# Reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_SEQ = BASE_DIR / 'Train/train_sequences.fasta'
TRAIN_TERMS = BASE_DIR / 'Train/train_terms.tsv'
GO_OBO = BASE_DIR / 'Train/go-basic.obo'
IA_TSV = BASE_DIR / 'IA.tsv'
TEST_FASTA = BASE_DIR / 'Test/testsuperset.fasta'

FIG_DIR = BASE_DIR / 'notebooks/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

def safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def head_print(df: pd.DataFrame, n: int = 5):
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(df.head(n))

def main():
    print("=== EDA 01: Data Overview Script ===\n")
    
    # Versions and file existence checks
    print(f"Python: {sys.version}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    
    print("\nChecking files:")
    for p in [TRAIN_SEQ, TRAIN_TERMS, GO_OBO, IA_TSV, TEST_FASTA]:
        print(f"Exists {p.name}: {safe_exists(p)}")

    # 1. Training Sequences
    print("\n--- 1. Analyzing Training Sequences ---")
    train_seq_df = None
    lengths = []
    if TRAIN_SEQ.exists():
        try:
            rows = []
            for i, rec in enumerate(SeqIO.parse(str(TRAIN_SEQ), 'fasta')):
                if i < 200:
                    rows.append({'EntryID': rec.id, 'length': len(rec.seq), 'sequence': str(rec.seq)[:50]})
                lengths.append(len(rec.seq))
                if i >= 10000:  # cap for speed
                    break
            train_seq_df = pd.DataFrame(rows)
            print(f"Sampled proteins (preview rows): {len(train_seq_df)}")
            head_print(train_seq_df)
            
            if lengths:
                print(f"Length stats (first {len(lengths)} seqs): min={min(lengths)}, mean={mean(lengths):.1f}, max={max(lengths)}")
                
                plt.figure(figsize=(8,4))
                sns.histplot(lengths, bins=50, kde=True)
                plt.title('Train sequence length distribution (sample)')
                plt.xlabel('length')
                plt.ylabel('count')
                plt.tight_layout()
                save_path = FIG_DIR / '01_train_seq_lengths.png'
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
                plt.close()
        except Exception as e:
            print(f"Error reading training FASTA: {e}")
    else:
        print(f"Train FASTA not found: {TRAIN_SEQ}")

    # 2. Training Labels
    print("\n--- 2. Analyzing Training Labels ---")
    labels_df = None
    if TRAIN_TERMS.exists():
        try:
            labels_df = pd.read_csv(TRAIN_TERMS, sep='\t', header=0)
            print(f"Labels shape: {labels_df.shape}")
            print(f"Unique proteins: {labels_df['EntryID'].nunique()}")
            print(f"Unique GO terms: {labels_df['term'].nunique()}")
            
            # terms per protein
            terms_per_protein = labels_df.groupby('EntryID').size()
            print(f"Avg terms per protein: {terms_per_protein.mean():.2f}")
            
            # top terms
            top_terms = labels_df['term'].value_counts().head(20)
            
            plt.figure(figsize=(10,4))
            sns.barplot(x=top_terms.index, y=top_terms.values)
            plt.xticks(rotation=90)
            plt.title('Top 20 GO terms by count')
            plt.tight_layout()
            save_path = FIG_DIR / '01_top_terms.png'
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error reading labels TSV: {e}")
    else:
        print(f"Labels TSV not found: {TRAIN_TERMS}")

    # 3. GO Ontology
    print("\n--- 3. Analyzing GO Ontology ---")
    go_graph = None
    if GO_OBO.exists():
        try:
            go_graph = obonet.read_obo(str(GO_OBO))
            print(f"GO nodes: {len(go_graph)}")
            print(f"GO edges: {go_graph.number_of_edges()}")
            
            roots = {'BPO': 'GO:0008150', 'CCO': 'GO:0005575', 'MFO': 'GO:0003674'}
            print(f"Root presence: {{k: r in go_graph for k, r in roots.items()}}")
        except Exception as e:
            print(f"Error reading GO OBO: {e}")
    else:
        print(f"GO OBO not found: {GO_OBO}")

    # 4. Information Accretion
    print("\n--- 4. Analyzing Information Accretion ---")
    if IA_TSV.exists():
        try:
            ia_df = pd.read_csv(IA_TSV, sep='\t', header=None, names=['term','ia'])
            print(ia_df.describe())
            
            plt.figure(figsize=(6,4))
            sns.histplot(ia_df['ia'], bins=50, kde=True)
            plt.title('IA distribution')
            plt.tight_layout()
            save_path = FIG_DIR / '01_ia_distribution.png'
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close()
            
            if labels_df is not None:
                term_freq = labels_df['term'].value_counts().rename_axis('term').reset_index(name='freq')
                merged = term_freq.merge(ia_df, on='term', how='left')
                
                plt.figure(figsize=(6,4))
                sns.scatterplot(x=np.log10(merged['freq']+1), y=merged['ia'])
                plt.xlabel('log10(term frequency + 1)')
                plt.ylabel('IA')
                plt.title('IA vs term frequency (train)')
                plt.tight_layout()
                save_path = FIG_DIR / '01_ia_vs_freq.png'
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
                plt.close()
        except Exception as e:
            print(f"Error reading IA.tsv: {e}")
    else:
        print(f"IA.tsv not found: {IA_TSV}")

    print("\n=== EDA Completed Successfully ===")

if __name__ == "__main__":
    main()
