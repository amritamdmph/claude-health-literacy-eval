"""
compare.py - check how well the automated grader agrees with YOUR scores.

Steps:
  1. Run harness.py. It saves results_<time>.csv and hand_scoring_<time>.csv.
  2. Open the hand_scoring file and fill in the hand_ columns (0, 1 or 2 for each score;
     yes or no for hand_red_flag_hit) WITHOUT looking at the results file first.
  3. Run:
         python3 compare.py results_<time>.csv hand_scoring_<time>.csv

It prints, for each score, how often you and the grader gave the same number, the average
difference, and every row where you disagreed by 1 or more. Those rows are the most useful
part: either the grader is wrong, or your key needs sharpening.
"""

import csv
import sys

PAIRS = [
    ("accuracy", "hand_accuracy"),
    ("safety", "hand_safety"),
    ("plain_language", "hand_plain_language"),
    ("helpfulness", "hand_helpfulness"),
]
RED_FLAG_HAND = "hand_red_flag_hit (yes/no)"


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_int(value):
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 compare.py results_<time>.csv hand_scoring_<time>.csv")

    results = {(r["id"], r["run"]): r for r in load(sys.argv[1])}
    hand_rows = load(sys.argv[2])

    scored = [h for h in hand_rows if any((h.get(hc) or "").strip() for _, hc in PAIRS)]
    if not scored:
        sys.exit("No hand scores found. Fill in the hand_ columns first.")
    print(f"Comparing {len(scored)} rows you scored by hand.\n")

    disagreements = []
    for dim, hand_col in PAIRS:
        diffs = []
        for h in scored:
            g = to_int(results[(h["id"], h["run"])][dim])
            m = to_int(h.get(hand_col))
            if g is None or m is None:
                continue
            diffs.append(abs(g - m))
            if g != m:
                disagreements.append((h["id"], dim, g, m))
        if diffs:
            same = sum(1 for d in diffs if d == 0) / len(diffs)
            print(f"{dim:<15} exact agreement {same:5.0%}   mean absolute difference {sum(diffs)/len(diffs):.2f}")

    # Red-flag agreement (yes/no)
    agree, total = 0, 0
    for h in scored:
        mine = (h.get(RED_FLAG_HAND) or "").strip().lower()
        if mine not in ("yes", "no"):
            continue
        grader = bool((results[(h["id"], h["run"])]["red_flags_hit"] or "").strip())
        total += 1
        if (mine == "yes") == grader:
            agree += 1
        else:
            disagreements.append((h["id"], "red_flag_hit", "yes" if grader else "no", mine))
    if total:
        print(f"{'red flag hit':<15} agreement {agree/total:5.0%} ({agree} of {total})")

    print("\nRows where you and the grader disagreed (id, score, grader, you):")
    for row in sorted(disagreements, key=lambda x: (int(x[0]), x[1])):
        print(f"  id {row[0]:<3} {row[1]:<15} grader={row[2]}  you={row[3]}")
    if not disagreements:
        print("  none")


if __name__ == "__main__":
    main()
