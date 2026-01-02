#!/usr/bin/env python3
"""
Extract the exact KNN prediction/scoring logic from both notebooks
to identify subtle differences in implementation.
"""
import json
import re
from pathlib import Path

def extract_function_bodies(code: str, func_names: list[str]) -> dict[str, str]:
    """Extract complete function definitions from code."""
    results = {}
    
    for func_name in func_names:
        # Match function definition to next function or end
        pattern = rf'(def {func_name}\([^)]*\):.*?)(?=\ndef\s|\nclass\s|\Z)'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            results[func_name] = match.group(1)
    
    return results

def extract_knn_scoring_block(code: str) -> str:
    """Extract the KNN neighbor scoring/aggregation logic."""
    # Look for patterns like:
    # - term_scores[term] += similarity
    # - score / max_score
    # - Counter() usage
    
    # Find blocks that contain neighbor iteration
    neighbor_blocks = []
    lines = code.split('\n')
    
    in_neighbor_loop = False
    current_block = []
    
    for line in lines:
        # Start capturing at neighbor loops
        if any(pattern in line.lower() for pattern in ['for.*neighbor', 'for.*nei_', 'kneighbors', 'nei_idx']):
            in_neighbor_loop = True
            current_block = [line]
        elif in_neighbor_loop:
            current_block.append(line)
            # End at dedent or new major section
            if line and not line[0].isspace() and not line.strip().startswith('#'):
                neighbor_blocks.append('\n'.join(current_block))
                in_neighbor_loop = False
                current_block = []
    
    if current_block:
        neighbor_blocks.append('\n'.join(current_block))
    
    return '\n\n--- NEIGHBOR SCORING BLOCK ---\n\n'.join(neighbor_blocks)

def analyze_notebook(nb_path: Path):
    """Extract all KNN-related implementation details."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    all_code = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                all_code.append(''.join(source))
            else:
                all_code.append(source)
    
    full_code = '\n\n'.join(all_code)
    
    print(f"{'='*80}")
    print(f"Analyzing: {nb_path.name}")
    print(f"{'='*80}\n")
    
    # 1. Extract embedding generation
    print("1. EMBEDDING GENERATION")
    print("-" * 40)
    emb_match = re.search(r'(def embed_sequences.*?)(?=\ndef\s|\Z)', full_code, re.DOTALL)
    if emb_match:
        print(emb_match.group(1)[:500])
    else:
        print("Not found in standard form")
    print()
    
    # 2. Extract KNN configuration
    print("2. KNN CONFIGURATION")
    print("-" * 40)
    knn_config = []
    for pattern in [
        r'NearestNeighbors\([^)]+\)',
        r'k\s*=\s*\d+',
        r'n_neighbors\s*=\s*\d+',
        r'metric\s*=\s*["\'][^"\']+["\']',
    ]:
        matches = re.findall(pattern, full_code)
        if matches:
            knn_config.extend(matches)
    
    for config in set(knn_config):
        print(f"  {config}")
    print()
    
    # 3. Extract neighbor scoring logic
    print("3. NEIGHBOR SCORING/AGGREGATION LOGIC")
    print("-" * 40)
    scoring = extract_knn_scoring_block(full_code)
    if scoring:
        print(scoring[:1500])  # First 1500 chars
    else:
        print("Not found")
    print()
    
    # 4. Extract normalization steps
    print("4. NORMALIZATION STEPS")
    print("-" * 40)
    norm_patterns = [
        r'(.*normalize.*)',
        r'(.*/ max_score.*)',
        r'(.*/ norm.*)',
        r'(.*/=.*std.*)',
    ]
    for pattern in norm_patterns:
        matches = re.findall(pattern, full_code, re.IGNORECASE)
        for match in matches[:5]:  # First 5 matches
            print(f"  {match.strip()}")
    print()
    
    # 5. Extract evaluation logic
    print("5. EVALUATION/THRESHOLD LOGIC")
    print("-" * 40)
    eval_match = re.search(r'(def evaluate.*?return [^}]+)', full_code, re.DOTALL)
    if eval_match:
        print(eval_match.group(1)[:800])
    else:
        # Look for inline evaluation
        threshold_blocks = re.findall(r'(threshold.*?f1.*?\n.*?\n.*?\n)', full_code, re.DOTALL | re.IGNORECASE)
        for block in threshold_blocks[:2]:
            print(block[:400])
    print()

def main():
    repo_root = Path(__file__).parent.parent
    
    nb1 = repo_root / 'notebooks' / '02_baseline_knn.ipynb'
    nb2 = repo_root / 'notebooks' / '05_cafa_e2e.ipynb'
    
    if nb1.exists():
        analyze_notebook(nb1)
        print("\n" * 3)
    
    if nb2.exists():
        analyze_notebook(nb2)

if __name__ == '__main__':
    main()
