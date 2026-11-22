"""Job A, step 1: mode classifier only (full-image decode vs the known message), per method.

Writes <out>/mode_<method>.json with the train-calibrated threshold and the full-image soft score of every train and
test image. job_a_finalize.py uses it as the fallback prediction (all ones if score >= thr, else all zeros) for any
image the frame search did not cover (crash, timeout). Takes a few minutes per method.

Run with PYTHONPATH=$PWD:$PWD/hack_test/experiments (it reuses frame_search.global_soft, so the scores match).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The script's own folder (experiments/, holding the OLD frame_search.py) comes first on sys.path: put hack_test ahead.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hack_test" / "experiments"))
from common import METHODS, best_threshold_acc, load_wm
from frame_search import global_soft
from wmloc.data import LocalisationDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="results_joba")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for method in a.methods:
        wm = load_wm(method)
        tr = LocalisationDataset(a.data, "train", methods=[method])
        tr_items = [tr[i] for i in range(len(tr))]
        g_tr = global_soft(wm, method, tr_items, True)
        fw = np.array([bool(it["full_watermark"]) for it in tr_items])
        acc, thr = best_threshold_acc(g_tr, fw)

        te = LocalisationDataset(a.data, "test", methods=[method])
        te_items = [te[i] for i in range(len(te))]
        g_te = global_soft(wm, method, te_items, True)

        res = dict(method=method, thr=thr, train_mode_acc=acc,
                   train={it["image_name"]: float(g) for it, g in zip(tr_items, g_tr)},
                   test={it["image_name"]: float(g) for it, g in zip(te_items, g_te)})
        (out / f"mode_{method}.json").write_text(json.dumps(res))
        print(f"[{method}] thr={thr:.4f} train mode acc={acc:.3f}; test: {len(te_items)} images, "
              f"{int((g_te >= thr).sum())} predicted fw=True", flush=True)


if __name__ == "__main__":
    main()
