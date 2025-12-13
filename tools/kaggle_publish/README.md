# Kaggle Dataset publishing (API-only)

These scripts stage artefacts and publish them as a Kaggle Dataset via the Kaggle API.

## 1) Get your Kaggle API token (one-off)

- Kaggle website → **Profile icon** → **Account** → **API** section → **Create New Token**
- This downloads a `kaggle.json` file.

Place it at:

- Windows: `C:\Users\<YOU>\.kaggle\kaggle.json`
- Linux/macOS: `~/.kaggle/kaggle.json`

Then ensure permissions are restrictive (Kaggle requires this on Linux/macOS):

- Linux/macOS: `chmod 600 ~/.kaggle/kaggle.json`

## 2) Install Kaggle API client

`pip install kaggle`

## 3) Stage Option B artefacts

From repo root:

`python tools/kaggle_publish/stage_option_b_artefacts.py --artefacts-dir artefacts_local/artefacts --out-dir kaggle_publish/cafa6_option_b`

## 4) Publish / version the dataset

`python tools/kaggle_publish/publish_dataset.py --dataset <username>/<dataset-slug> --title "CAFA6 Option B Artefacts" --dir kaggle_publish/cafa6_option_b --message "v1: priors + tfidf"`

## Kaggle-side usage

- Attach the dataset to your Kaggle notebook.
- Set environment variable `CAFA_OPTION_B_ARTEFACTS_DIR` to the dataset folder if you want explicit pathing.

Notes:
- Publishing requires authentication; there is no truly "zero manual" way because Kaggle must issue the token.
- Do **not** commit `kaggle.json` to git.
