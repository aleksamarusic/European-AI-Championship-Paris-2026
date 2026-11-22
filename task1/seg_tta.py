"""Test-time augmentation for a trained segmenter run: average the probabilities over the 8 flips/rotations (D4)
of each image, then pick the threshold on the run's real validation images (same THRESHOLDS as seg_train).
Writes pseudo-run results_seg/<tag>_tta (pred_val / pred_test probabilities, metrics.json) so seg_combine.py
compares it with every other run on the same validation images.

    python experiments/seg_tta.py --tag tm_fresh_band2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_train import UNet, evaluate, group_of, load_items  # noqa: E402


@torch.no_grad()
def predict_d4(model, images, bs=8, rots=4):
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).cuda().permute(0, 3, 1, 2).float() / 255
        acc = 0
        for flip in (False, True):
            xf = x.flip(3) if flip else x
            for k in range(rots):
                p = torch.sigmoid(model(torch.rot90(xf, k, (2, 3))))
                p = torch.rot90(p, -k, (2, 3))
                acc = acc + (p.flip(3) if flip else p)
        out.append((acc / (2 * rots))[:, 0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips", action="store_true", help="horizontal flip only (2 views) instead of all 8 D4 views")
    ap.add_argument("--tag", required=True); ap.add_argument("--seg", default="results_seg"); ap.add_argument("--data", default="release")
    a = ap.parse_args()
    run = Path(a.seg) / a.tag
    met = json.loads((run / "metrics.json").read_text()); args = met["args"]
    model = UNet(args["width"], args.get("inp", "hp3")).cuda()
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    tr = {it["image_name"]: it for it in load_items(a.data, "train", [met["method"]])}
    val = [tr[n] for n in met["val_names"]]
    rots = 1 if a.flips else 4; suffix = "_ttaflip" if a.flips else "_tta"
    pv = predict_d4(model, [it["image"] for it in val], rots=rots)
    thr, miou, by_group, sweep = evaluate(pv, [it["mask128"] for it in val], [group_of(it) for it in val])
    print(f"[{a.tag}] validation: plain {met['val_miou']:.4f} @thr {met['thr']} -> TTA({2 * rots} views) {miou:.4f} @thr {thr} {by_group}", flush=True)
    te = load_items(a.data, "test", [met["method"]])
    pt = predict_d4(model, [it["image"] for it in te], rots=rots)
    out = Path(a.seg) / f"{a.tag}{suffix}"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "pred_val.npz", names=np.array([it["image_name"] for it in val]), probs=(pv * 255).round().astype(np.uint8))
    np.savez_compressed(out / "pred_test.npz", names=np.array([it["image_name"] for it in te]), probs=(pt * 255).round().astype(np.uint8))
    (out / "metrics.json").write_text(json.dumps(dict(met, tag=f"{a.tag}{suffix}", val_miou=float(miou), thr=float(thr),
                                                     by_group=by_group, thr_sweep=sweep, tta="flip" if a.flips else "d4"), indent=1))
    print(f"[{a.tag}] wrote {out}")


if __name__ == "__main__":
    main()
