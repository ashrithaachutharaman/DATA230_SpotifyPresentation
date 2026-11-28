"""Day-1 check: rebuild the model table + split from the cleaned file and print the counts
and a short hash of the test track ids.

Run from the repo root:   python -m src.check_setup
Everyone should see identical output, including the test hash. If not, someone has a different
tracks_clean.csv or different package versions (see requirements.txt).
If data/split.csv exists (the shared split from notebook 01), its hash is checked too.
"""
from src.style import *  # noqa: F401,F403

if __name__ == "__main__":
    model_df = make_model_table(load_clean())
    split_df = make_split(model_df)
    check_split(split_df)

    if SPLIT_PATH.exists():
        saved = pd.read_csv(SPLIT_PATH)
        saved_hash, rebuilt_hash = split_hash(saved), split_hash(split_df)
        print(f"\n{SPLIT_PATH.name} test hash: {saved_hash}")
        if saved_hash == rebuilt_hash:
            print("OK: rebuilt split matches split.csv")
        else:
            print("MISMATCH: rebuilt split differs from split.csv (check data file and package versions)")
