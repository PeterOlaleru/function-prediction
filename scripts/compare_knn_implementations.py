#!/usr/bin/env python3
"""
Diagnostic script to compare KNN implementations in two notebooks:
- 02_baseline_knn.ipynb (older, good F1 > 0.216)
- 05_cafa_e2e.ipynb (newer, underperforming)

Extracts and compares:
- Embedding model used
- KNN parameters (k, metric, etc.)
- Score aggregation logic
- Normalization approaches
- Evaluation methodology
"""
import json
import re
from pathlib import Path
from typing import Any

def extract_code_cells(nb_path: Path) -> list[str]:
    """Extract all code cells from a notebook."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code_cells.append(''.join(source))
            else:
                code_cells.append(source)
    return code_cells

def find_patterns(code_cells: list[str], patterns: dict[str, str]) -> dict[str, list[str]]:
    """Find all matches for given regex patterns in code cells."""
    results = {name: [] for name in patterns}
    
    for cell in code_cells:
        for name, pattern in patterns.items():
            matches = re.findall(pattern, cell, re.MULTILINE | re.IGNORECASE)
            if matches:
                results[name].extend(matches)
    
    return results

def analyze_notebook(nb_path: Path) -> dict[str, Any]:
    """Analyze a notebook and extract KNN-related configurations."""
    code_cells = extract_code_cells(nb_path)
    all_code = '\n'.join(code_cells)
    
    patterns = {
        'embedding_model': r'model_name\s*=\s*["\']([^"\']+)["\']',
        'esm_model': r'(esm2?[_-]?t?\d+[_-]?\d+[MBG]?[_-]?[A-Z0-9]*)',
        'knn_neighbors': r'(?:k|n_neighbors)\s*=\s*(\d+)',
        'knn_metric': r'metric\s*=\s*["\']([^"\']+)["\']',
        'normalize': r'normalize|normali[sz]ed?',
        'distance_weight': r'(?:1\s*-\s*distance|similarity)',
        'score_aggregation': r'(?:term_scores|Counter\(\)|\.update)',
        'max_score': r'max_score|max\(.*scores',
        'threshold': r'threshold\s*=\s*([0-9.]+)',
    }
    
    results = find_patterns(code_cells, patterns)
    
    # Additional analysis
    analysis = {
        'notebook': str(nb_path.name),
        'embedding_model': results['embedding_model'],
        'esm_model': results['esm_model'],
        'knn_neighbors': results['knn_neighbors'],
        'knn_metric': results['knn_metric'],
        'uses_normalize': len(results['normalize']) > 0,
        'distance_weighting': len(results['distance_weight']) > 0,
        'score_aggregation_found': len(results['score_aggregation']) > 0,
        'max_score_normalization': len(results['max_score']) > 0,
        'thresholds': results['threshold'],
        'code_length': len(all_code),
        'num_cells': len(code_cells),
    }
    
    # Look for specific embedding dimension patterns
    emb_dim_matches = re.findall(r'embeddings.*shape.*\((\d+),\s*(\d+)\)', all_code)
    if emb_dim_matches:
        analysis['embedding_dims'] = emb_dim_matches
    
    # Check for L2 normalization
    l2_norm = re.findall(r'l2_normalize|normalize.*l2|embeddings\s*/=.*norm', all_code, re.IGNORECASE)
    analysis['l2_normalization'] = len(l2_norm) > 0
    
    return analysis

def compare_notebooks(nb1_path: Path, nb2_path: Path):
    """Compare two notebooks and print differences."""
    print("="*80)
    print("KNN IMPLEMENTATION COMPARISON")
    print("="*80)
    print()
    
    nb1_analysis = analyze_notebook(nb1_path)
    nb2_analysis = analyze_notebook(nb2_path)
    
    print(f"📓 NOTEBOOK 1: {nb1_analysis['notebook']}")
    print(f"   Code cells: {nb1_analysis['num_cells']}")
    print(f"   Total code length: {nb1_analysis['code_length']:,} chars")
    print()
    
    print(f"📓 NOTEBOOK 2: {nb2_analysis['notebook']}")
    print(f"   Code cells: {nb2_analysis['num_cells']}")
    print(f"   Total code length: {nb2_analysis['code_length']:,} chars")
    print()
    print("="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    print()
    
    # Compare each aspect
    aspects = [
        ('Embedding Model', 'embedding_model'),
        ('ESM Model Found', 'esm_model'),
        ('KNN Neighbors (k)', 'knn_neighbors'),
        ('KNN Metric', 'knn_metric'),
        ('Uses Normalization', 'uses_normalize'),
        ('L2 Normalization', 'l2_normalization'),
        ('Distance Weighting', 'distance_weighting'),
        ('Score Aggregation', 'score_aggregation_found'),
        ('Max Score Normalization', 'max_score_normalization'),
        ('Thresholds', 'thresholds'),
    ]
    
    for label, key in aspects:
        val1 = nb1_analysis.get(key)
        val2 = nb2_analysis.get(key)
        
        if val1 != val2:
            print(f"⚠️  {label}:")
            print(f"   Notebook 1: {val1}")
            print(f"   Notebook 2: {val2}")
            print()
        else:
            print(f"✅ {label}: {val1}")
            print()
    
    if 'embedding_dims' in nb1_analysis or 'embedding_dims' in nb2_analysis:
        print("📊 Embedding Dimensions:")
        print(f"   Notebook 1: {nb1_analysis.get('embedding_dims', 'Not found')}")
        print(f"   Notebook 2: {nb2_analysis.get('embedding_dims', 'Not found')}")
        print()

def main():
    repo_root = Path(__file__).parent.parent
    nb1 = repo_root / 'notebooks' / '02_baseline_knn.ipynb'
    nb2 = repo_root / 'notebooks' / '05_cafa_e2e.ipynb'
    
    if not nb1.exists():
        print(f"ERROR: {nb1} not found")
        return 1
    
    if not nb2.exists():
        print(f"ERROR: {nb2} not found")
        return 1
    
    compare_notebooks(nb1, nb2)
    return 0

if __name__ == '__main__':
    exit(main())
