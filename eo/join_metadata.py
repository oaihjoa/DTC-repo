"""Eedi student metadata -> per-uid gender groups json (for eo/).

INPUT   data/nips_task34/student_metadata_task_3_4.csv  (raw Eedi release,
        arXiv:2007.12061 Sec. 2.3: UserId, Gender 0=unspec/1=f/2=m/3=other),
        and data/nips_task34/keyid2idx.json (pyKT's original->index uid map).
OUTPUT  data/groups_nips_task34.json:  {"gender": {uid_index: 1|2|-1}}
        (-1 = excluded: Gender 0/3; excluded students still train BCE/KL —
        they are outside the gap term only.)

Run:  python eo/join_metadata.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/nips_task34/student_metadata_task_3_4.csv"
KEYID = ROOT / "data/nips_task34/keyid2idx.json"
OUT = ROOT / "data/groups_nips_task34.json"


def main():
    assert META.exists(), f"place the raw Eedi metadata csv at {META}"
    assert KEYID.exists(), f"place pyKT's keyid2idx.json at {KEYID}"
    # pyKT re-indexes students; map raw Eedi UserId -> pyKT uid index so the
    # labels line up with the uid column of train_valid_sequences.csv
    uid_map = {str(k): int(v)
               for k, v in json.loads(KEYID.read_text())["uid"].items()}
    gender = {}
    for _, row in pd.read_csv(META).iterrows():
        idx = uid_map.get(str(row["UserId"]))
        if idx is None:
            # student never appears in the pyKT split — no window to label
            continue
        # keep only binary gender codes 1/2; 0 (unspecified) and 3 (other)
        # are excluded from the gap term via -1 (too few rows to estimate)
        gender_code = int(row["Gender"]) if pd.notna(row["Gender"]) else 0
        gender[idx] = gender_code if gender_code in (1, 2) else -1
    OUT.write_text(json.dumps({"gender": gender}))
    print(f"gender: {sum(1 for v in gender.values() if v > 0)} labeled "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
