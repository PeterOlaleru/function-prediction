import pandas as pd
import obonet
import networkx as nx
from collections import Counter
import os

# Paths
TRAIN_TERMS_PATH = r'c:\Users\Olale\Documents\Codebase\Science\cafa-6-protein-function-prediction\Train\train_terms.tsv'
OBO_PATH = r'c:\Users\Olale\Documents\Codebase\Science\cafa-6-protein-function-prediction\Train\go-basic.obo'

def load_obo(path):
    print(f"Loading OBO from {path}...")
    graph = obonet.read_obo(path)
    term_to_namespace = {}
    for node, data in graph.nodes(data=True):
        if 'namespace' in data:
            # Map to short codes
            ns = data['namespace']
            if ns == 'biological_process':
                term_to_namespace[node] = 'BP'
            elif ns == 'molecular_function':
                term_to_namespace[node] = 'MF'
            elif ns == 'cellular_component':
                term_to_namespace[node] = 'CC'
    return term_to_namespace

def analyze_terms():
    if not os.path.exists(TRAIN_TERMS_PATH):
        print(f"Error: {TRAIN_TERMS_PATH} not found.")
        return

    if not os.path.exists(OBO_PATH):
        print(f"Error: {OBO_PATH} not found.")
        return

    # Load OBO
    term_ns = load_obo(OBO_PATH)
    
    # Load Train Terms
    print(f"Loading train terms from {TRAIN_TERMS_PATH}...")
    # train_terms.tsv columns: EntryID, term, aspect (sometimes aspect is in the file, let's check)
    # Usually: EntryID \t term \t aspect
    df = pd.read_csv(TRAIN_TERMS_PATH, sep='\t')
    print(f"Loaded {len(df)} rows.")
    
    # Check columns
    print(f"Columns: {df.columns.tolist()}")
    
    # Count terms
    term_counts = df['term'].value_counts()
    print(f"Total unique terms in training: {len(term_counts)}")
    
    # Analyze by aspect
    aspect_counts = {'BP': [], 'MF': [], 'CC': []}
    unknown_aspect = []
    
    for term, count in term_counts.items():
        ns = term_ns.get(term)
        if ns:
            aspect_counts[ns].append((term, count))
        else:
            unknown_aspect.append((term, count))
            
    print(f"Terms with unknown aspect: {len(unknown_aspect)}")
    
    # Stats per aspect
    print("\n--- Analysis per Aspect ---")
    total_annotations = len(df)
    
    proposed_cutoffs = {
        'BP': 10000,
        'MF': 2000,
        'CC': 1500
    }
    
    for ns in ['BP', 'MF', 'CC']:
        terms = aspect_counts[ns]
        # Sort by count desc
        terms.sort(key=lambda x: x[1], reverse=True)
        
        n_unique = len(terms)
        total_ns_annotations = sum(c for t, c in terms)
        
        print(f"\nAspect: {ns}")
        print(f"  Unique Terms: {n_unique}")
        print(f"  Total Annotations: {total_ns_annotations}")
        
        # Check coverage of proposed cutoff
        cutoff = proposed_cutoffs[ns]
        top_k_terms = terms[:cutoff]
        top_k_annotations = sum(c for t, c in top_k_terms)
        coverage = (top_k_annotations / total_ns_annotations) * 100 if total_ns_annotations > 0 else 0
        min_freq = top_k_terms[-1][1] if top_k_terms else 0
        
        print(f"  Proposed Top-{cutoff}:")
        print(f"    Coverage of annotations: {coverage:.2f}%")
        print(f"    Minimum frequency included: {min_freq}")
        
        # Suggest optimal cutoff for 99% coverage
        current_sum = 0
        target_99 = 0.99 * total_ns_annotations
        count_99 = 0
        for i, (t, c) in enumerate(terms):
            current_sum += c
            if current_sum >= target_99:
                count_99 = i + 1
                break
        print(f"  Terms needed for 99% coverage: {count_99}")

if __name__ == "__main__":
    analyze_terms()
