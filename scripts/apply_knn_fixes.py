#!/usr/bin/env python3
"""
Apply KNN performance fixes to 05_cafa_e2e.ipynb

This script automatically applies the three critical fixes:
1. Reduce k from 50 to 10
2. Remove IA weighting from neighbor aggregation
3. Add per-protein max normalization

Creates a backup before modifying.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

def apply_fixes(nb_path: Path, output_path: Path = None) -> bool:
    """Apply all KNN fixes to the notebook."""
    
    if not nb_path.exists():
        print(f"ERROR: {nb_path} not found")
        return False
    
    # Create backup
    backup_path = nb_path.parent / f"{nb_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    shutil.copy2(nb_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Load notebook
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and fix KNN cell
    fixed = False
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            
            if '# CELL 13E - KNN' in code:
                print(f"\n📍 Found KNN cell at index {i}")
                original_code = code
                
                # Apply Fix 1: Reduce k from 50 to 10
                if "KNN_K = int(globals().get('KNN_K', 50))" in code:
                    code = code.replace(
                        "KNN_K = int(globals().get('KNN_K', 50))",
                        "KNN_K = int(globals().get('KNN_K', 10))  # FIXED: reduced from 50 to 10 (matches baseline)"
                    )
                    print("✅ Fix 1: Reduced k from 50 to 10")
                
                # Apply Fix 2a: Remove IA weighting in OOF predictions
                if "* Y_nei * w_ia_broadcast).sum(axis=1) / denom" in code:
                    code = code.replace(
                        "Y_nei = Y_knn[neigh_b]  # (B, K, L)\n                    # IA-weighted aggregation: sims @ (Y_nei * w_ia)\n                    scores = ((sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom).astype(np.float32)",
                        "Y_nei = Y_knn[neigh_b]  # (B, K, L)\n                    # FIXED: Removed IA weighting during aggregation (apply only in evaluation)\n                    # Simple similarity-weighted average (matches baseline logic)\n                    scores = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1).astype(np.float32)"
                    )
                    print("✅ Fix 2a: Removed IA weighting from OOF predictions")
                
                # Apply Fix 2b: Remove IA weighting in test predictions
                if "* Y_nei * w_ia_broadcast).sum(axis=1) / denom_te[i:j]" in code:
                    code = code.replace(
                        "Y_nei = Y_knn[neigh_b]\n                scores = ((sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1) / denom_te[i:j]).astype(np.float32)",
                        "Y_nei = Y_knn[neigh_b]\n                # FIXED: Removed IA weighting during aggregation\n                scores = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1).astype(np.float32)"
                    )
                    print("✅ Fix 2b: Removed IA weighting from test predictions")
                
                # Apply Fix 3: Add per-protein max normalization
                # Insert before "# RANK-1: Finite-value quality gates"
                normalization_code = """
        # CRITICAL FIX: Per-protein max normalization (matches baseline logic)
        # This ensures each protein's scores are calibrated to [0, 1] range
        print('[KNN] Applying per-protein max normalization...')
        
        # Normalize OOF predictions
        for i in range(oof_pred_knn.shape[0]):
            max_val = oof_pred_knn[i].max()
            if max_val > 1e-9:  # Avoid division by zero
                oof_pred_knn[i] /= max_val
        
        # Normalize test predictions  
        for i in range(test_pred_knn.shape[0]):
            max_val = test_pred_knn[i].max()
            if max_val > 1e-9:
                test_pred_knn[i] /= max_val
        
        print(f'[KNN] Normalization complete. OOF score range: [{oof_pred_knn.min():.4f}, {oof_pred_knn.max():.4f}]')
        print(f'[KNN] Normalization complete. Test score range: [{test_pred_knn.min():.4f}, {test_pred_knn.max():.4f}]')
        
"""
                if "# RANK-1: Finite-value quality gates" in code and "[KNN] Applying per-protein max normalization" not in code:
                    code = code.replace(
                        "        # RANK-1: Finite-value quality gates",
                        normalization_code + "        # RANK-1: Finite-value quality gates"
                    )
                    print("✅ Fix 3: Added per-protein max normalization")
                
                # Update cell source
                if code != original_code:
                    cell['source'] = code.split('\n')
                    # Add trailing newline to last line if needed
                    if cell['source'] and not cell['source'][-1].endswith('\n'):
                        cell['source'][-1] += '\n'
                    fixed = True
                    print(f"\n✅ Applied {3} fixes to cell {i}")
                else:
                    print(f"\n⚠️  No changes made (fixes may already be applied or cell format differs)")
                
                break
    
    if not fixed:
        print("\n❌ KNN cell not found or fixes could not be applied")
        return False
    
    # Save modified notebook
    output_path = output_path or nb_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n✅ Saved fixed notebook to: {output_path}")
    print(f"📋 Backup saved to: {backup_path}")
    
    return True

def main():
    repo_root = Path(__file__).parent.parent
    nb_path = repo_root / 'notebooks' / '05_cafa_e2e.ipynb'
    
    print("="*80)
    print("KNN Performance Fix Applicator")
    print("="*80)
    print(f"\nTarget notebook: {nb_path}")
    print("\nThis will apply 3 critical fixes:")
    print("  1. Reduce k from 50 to 10")
    print("  2. Remove IA weighting from neighbor aggregation")
    print("  3. Add per-protein max normalization")
    print()
    
    success = apply_fixes(nb_path)
    
    if success:
        print("\n" + "="*80)
        print("SUCCESS! Fixes applied.")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review the changes in the notebook")
        print("  2. Run the KNN cell (Cell 13E)")
        print("  3. Verify F1 score > 0.216")
        print("  4. Compare with baseline performance")
        return 0
    else:
        print("\n" + "="*80)
        print("FAILED to apply fixes")
        print("="*80)
        print("\nPlease apply fixes manually using docs/KNN_FIX_IMPLEMENTATION.md")
        return 1

if __name__ == '__main__':
    exit(main())
