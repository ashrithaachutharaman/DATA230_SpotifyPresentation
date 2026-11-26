# DATA230 Spotify Presentation — Can a Song's Sound Tell Us Its Era?

## Repo layout
```
data/         tracks_clean.csv (input), model_data.parquet, split.csv   (Jason)   — not in git
outputs/      predictions.csv, cv_scores.csv, model.joblib, tsne_coords.csv,
              pca_scores.csv, importance.csv, shap_values.npy
src/style.py  LOCKED: SEED, FEATURES, ERA_LABELS, ERA_COLORS, make_model_table, make_split
notebooks/    01_data_split (Jason), 02_models (Ashritha), 03_explain (Peter)
dashboards/   dash_app.py (Jason), streamlit_app.py (Ashritha)
```

## Day-1 setup (everyone)
1. `pip install -r requirements.txt`
2. Run the cleaning section of `analysis.ipynb` (needs Kaggle `tracks.csv` + `artists.csv`); it writes `data/tracks_clean.csv`.
   Or download the shared `tracks_clean.csv` into `data/`.
3. From the repo root: `python -m src.check_setup`
4. Post your counts in the group chat. They must match exactly; if not, you have a different cleaned file.

In a notebook: `from src.style import *` (the stubs in `notebooks/` already add the repo root to the path).

Change anything in `src/style.py` only with the whole group.
