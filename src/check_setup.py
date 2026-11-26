"""Day-1 check: rebuild the model table + split from the cleaned file and print the counts.

Run from the repo root:   python -m src.check_setup
Everyone should see identical numbers. If not, someone has a different tracks_clean.csv.
"""
from src.style import *  # noqa: F401,F403

if __name__ == "__main__":
    model_df = make_model_table(load_clean())
    split_df = make_split(model_df)
    check_split(split_df)
