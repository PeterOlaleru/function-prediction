# 🧬 Protein Function Prediction - Simple Explainer

## What Are We Building?

**Think of proteins like LEGO blocks that make up life.** Each protein has a specific job in your body.

This project is like building a **"smart detective"** that can:
- Look at a protein's amino acid sequence (its building instructions)
- Predict what job that protein does
- Tell us where it works in a cell
- Describe its function in plain English

---

## Real World Example

Imagine you have a mystery protein made of amino acids:
```
MKALIVLGL...QWERTY... (hundreds of letters)
```

Your model will predict:
- **What it does**: "This protein binds DNA"
- **Where it works**: "Inside the cell nucleus"
- **What process**: "Helps repair damaged DNA"

---

## The Challenge in Simple Terms

### What You Have:
1. **Training Data**: 
   - Protein sequences (like MKALIVLGL...)
   - Labels showing what each protein does
   - A map of how functions relate to each other (Gene Ontology)

2. **Test Data**: 
   - New protein sequences
   - NO LABELS (you predict these!)

### What Makes This Hard:
- One protein can do MANY jobs (multi-label problem)
- Some functions are rare (imbalanced data)
- Functions are organized in a hierarchy (tree structure)
- You won't know if you're right until months later (prospective evaluation)

---

## What Success Looks Like

You submit predictions like:
```
Protein_A    GO:0003677    0.95    (DNA binding, 95% confident)
Protein_A    GO:0005634    0.87    (In nucleus, 87% confident)
Protein_B    GO:0016887    0.73    (Helps move stuff, 73% confident)
```

Your score is based on:
- **Precision**: How many predictions were correct?
- **Recall**: Did you catch all the important functions?
- **Weighted F1**: Harder-to-predict functions count more

---

## Key Vocabulary (Simple)

- **Amino Acid Sequence**: The "recipe" for building a protein (20 letter alphabet)
- **GO Term**: A label describing what a protein does (like GO:0003677 = "DNA binding")
- **Subontology**: Three categories - what it does (MF), which process (BP), where it is (CC)
- **FASTA File**: Text file with protein sequences
- **Information Accretion**: Weight showing how hard a function is to predict (rare = higher weight)

---

## The Three Main Categories

1. **Molecular Function (MF)**: What the protein does at molecular level
   - Example: "binds to DNA" or "cuts proteins"

2. **Biological Process (BP)**: What bigger process it helps with
   - Example: "DNA repair" or "cell division"

3. **Cellular Component (CC)**: Where in the cell it works
   - Example: "nucleus" or "cell membrane"

---

## Why This Matters

**Medical Impact**: 
- Finding proteins that cause diseases
- Designing new drugs
- Understanding how cells work
- Speeding up research (instead of years in lab, AI predicts in seconds)

---

## Bottom Line

You're building an AI that reads protein "recipes" and predicts their jobs in the cell. 

**Think**: "Google Translate" but for proteins → functions instead of English → Spanish.

---

## Data Files Quick Reference

| File | What It Contains |
|------|------------------|
| `train_sequences.fasta` | Training protein sequences |
| `train_terms.tsv` | Labels for training (what each does) |
| `go-basic.obo` | Function hierarchy map |
| `testsuperset.fasta` | Proteins you need to predict |
| `IA.tsv` | Difficulty weights for each function |
| `sample_submission.tsv` | Example of what to submit |

---

## Next Steps

Read the `PLAN.md` file for the detailed step-by-step building plan!
