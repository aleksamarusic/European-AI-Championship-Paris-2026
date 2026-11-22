"""Run 8-view D4 TTA (or 4-view H+V flip TTA) on a trained ps_convnext_fpn checkpoint."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from ps_convnext_fpn import FrozenMsgExtractor, PixelSealFPN, eval_mode_aware  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


@torch.no_grad()
def predict_d4(model, fext, items, bs=4, rots=4):
    model.eval()
    probs = []
    for i in range(0, len(items), bs):
        sub = items[i:i + bs]
        x = torch.from_numpy(np.stack([it["image"] for it in sub])).cuda().permute(0, 3, 1, 2).float() / 255.0
        s = torch.from_numpy(np.stack([2.0 * it["message"].astype(np.float32) - 1.0 for it in sub])).cuda()
        acc = 0
        for flip in (False, True):
            xf = x.flip(3) if flip else x
            for k in range(rots):
                xr = torch.rot90(xf, k, (2, 3))
                m16, a128 = fext(xr, s)
                p = torch.sigmoid(model(xr, m16, a128))
                p = torch.rot90(p, -k, (2, 3))
                acc = acc + (p.flip(3) if flip else p)
        probs.append((acc / (2 * rots))[:, 0].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ps_convnext_fpn")
    ap.add_argument("--out_tag", default="ps_convnext_fpn_d4")
    ap.add_argument("--rots", type=int, default=4)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wm = load_watermark("pixelseal", device="cuda")
    fext = FrozenMsgExtractor(wm)
    model = PixelSealFPN(copy.deepcopy(wm.det.convnext).cuda()).cuda()
    model.load_state_dict(torch.load(ROOT / "results_seg" / a.tag / "model.pt", weights_only=True))

    val_names = set(json.loads((ROOT / "results_seg" / "pixelseal" / "metrics.json").read_text())["val_names"])
    tr = LocalisationDataset(ROOT / "release", "train", methods=["pixelseal"])
    val_real = sorted([tr[i] for i in range(len(tr)) if tr.meta[i]["image_name"] in val_names],
                      key=lambda it: it["image_name"])
    mode_json = json.loads((ROOT / "results_joba" / "mode_pixelseal_fp32.json").read_text())
    val_fw = [bool(mode_json["train"][it["image_name"]] >= mode_json["thr"]) for it in val_real]
    val_gts = [it["mask128"] for it in val_real]

    pv = predict_d4(model, fext, val_real, bs=4, rots=a.rots)
    m_mode, (tT, tF), sT, sF, b_sing, m_sing = eval_mode_aware(pv, val_gts, val_fw)
    print(f"[{a.out_tag}] D4(rots={a.rots}) val: single {m_sing:.4f}@{b_sing} | mode-thr {m_mode:.4f}@({tT},{tF}) [fwT {sT:.3f} fwF {sF:.3f}]", flush=True)

    te_ds = LocalisationDataset(ROOT / "release", "test", methods=["pixelseal"])
    te_items = [te_ds[i] for i in range(len(te_ds))]
    pt = predict_d4(model, fext, te_items, bs=4, rots=a.rots)

    out = ROOT / "results_seg" / a.out_tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "pred_val.npz", names=np.array([it["image_name"] for it in val_real]),
                        probs=(pv * 255.0).round().astype(np.uint8))
    np.savez_compressed(out / "pred_test.npz", names=np.array([it["image_name"] for it in te_items]),
                        probs=(pt * 255.0).round().astype(np.uint8))
    (out / "metrics.json").write_text(json.dumps(dict(
        method="pixelseal", tag=a.out_tag, val_miou=float(m_sing), val_miou_mode=float(m_mode),
        thr=float(b_sing), thresholds=[tT, tF], by_group={"fw=True": sT, "fw=False": sF},
        perfect_mode_val=0.4499, val_names=sorted(val_names)), indent=1))
    print(f"[{a.out_tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
