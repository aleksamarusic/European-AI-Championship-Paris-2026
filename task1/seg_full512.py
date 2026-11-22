"""Full-512x512 Soft-mIoU + zero-init Mode-FiLM fine-tuning for TrustMark and StegaStamp segmenters.

Fixes three major bottlenecks in seg_train.py:
  1. Full 512x512 training (instead of 256x256 crops) so global context is visible.
  2. 0.4*BCE + 0.6*SymmetricSoftIoULoss (directly optimizes 0.5*(IoU_0 + IoU_1), fixing small holes on fw=True!).
  3. Zero-initialized Mode-FiLM conditioning from the known full-image decode score (eliminating mode confusion
     like TrustMark's 3c541165b889 while preserving 100% of the pretrained checkpoint at step 0!).
  4. Pure clean COCO covers (no cross-method StegaStamp noise contamination).
Runs TTA automatically at the end and writes continuous probabilities for final ensembling.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from seg_train import UNet, embed_full, embed_rect, evaluate, group_of, is_rect, load_items  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def sym_soft_iou_loss(logits, target):
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter1 = (p * target).sum(dims)
    union1 = (p + target - p * target).sum(dims)
    iou1 = (inter1 + 1.0) / (union1 + 1.0)
    inter0 = ((1.0 - p) * (1.0 - target)).sum(dims)
    union0 = ((1.0 - p) + (1.0 - target) - (1.0 - p) * (1.0 - target)).sum(dims)
    iou0 = (inter0 + 1.0) / (union0 + 1.0)
    return (1.0 - 0.5 * (iou1 + iou0)).mean()


class ModeConditionedUNet(nn.Module):
    """Wraps pretrained UNet(w, inp) with zero-initialized Mode-FiLM biases at e4 and d2."""

    def __init__(self, base_unet: UNet):
        super().__init__()
        self.base = base_unet
        c4 = base_unet.e4[0].out_channels
        c2 = base_unet.d2[0].out_channels
        self.m4 = nn.Linear(2, c4)
        self.m2 = nn.Linear(2, c2)
        nn.init.zeros_(self.m4.weight); nn.init.zeros_(self.m4.bias)
        nn.init.zeros_(self.m2.weight); nn.init.zeros_(self.m2.bias)

    def forward(self, x, mode_feat):
        # mode_feat: [B, 2] = [mode_score, float(mode_score >= thr)]
        u = self.base
        if u.inp == "luma":
            z = torch.cat([(x - 0.5) * 4, u.band(x.mean(1, keepdim=True))], 1)
        elif u.band is not None:
            z = torch.cat([(x - 0.5) * 4, u.band(x)], 1)
        else:
            hp = x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
            z = torch.cat([(x - 0.5) * 4, hp * 20], 1)
        e0 = u.e0(z)
        e1 = u.e1(F.max_pool2d(e0, 2))
        e2 = u.e2(F.max_pool2d(e1, 2))
        e3 = u.e3(F.max_pool2d(e2, 2))
        e4 = u.e4(F.max_pool2d(e3, 2)) + self.m4(mode_feat)[:, :, None, None]
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        d3 = u.d3(torch.cat([up(e4), e3], 1))
        d2 = u.d2(torch.cat([up(d3), e2], 1)) + self.m2(mode_feat)[:, :, None, None]
        return u.head(d2)


@torch.no_grad()
def predict_tta(model, images, mode_feats, bs=8, rots=1):
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).cuda().permute(0, 3, 1, 2).float() / 255.0
        mf = torch.from_numpy(np.stack(mode_feats[i:i + bs])).cuda().float()
        acc = 0
        for flip in (False, True):
            xf = x.flip(3) if flip else x
            for k in range(rots):
                p = torch.sigmoid(model(torch.rot90(xf, k, (2, 3)), mf))
                p = torch.rot90(p, -k, (2, 3))
                acc = acc + (p.flip(3) if flip else p)
        out.append((acc / (2 * rots))[:, 0].cpu().numpy())
    model.train()
    return np.concatenate(out)


def mode_feat_of(score, thr):
    if score is None or math.isnan(score):
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([float(score), float(score >= thr)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["trustmark", "stegastamp"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--minutes", type=float, default=26.0)
    ap.add_argument("--eval_every", type=float, default=2.5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_coco", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    t_start = time.time()
    out = ROOT / "results_seg" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    tr_all = load_items(ROOT / "release", "train", ("pixelseal", "trustmark", "stegastamp"))
    val_names = set(json.loads((ROOT / "results_seg" / a.method / "metrics.json").read_text())["val_names"])
    real = [it for it in tr_all if it["method"] == a.method]
    train_real = [it for it in real if it["image_name"] not in val_names]
    val_real = sorted([it for it in real if it["image_name"] in val_names], key=lambda it: it["image_name"])

    mode_json = json.loads((ROOT / "results_joba" / f"mode_{a.method}.json").read_text())
    thr_mode = mode_json["thr"]
    val_imgs = [it["image"] for it in val_real]
    val_gts = [it["mask128"] for it in val_real]
    val_groups = [group_of(it) for it in val_real]
    val_mf = [mode_feat_of(mode_json["train"][it["image_name"]], thr_mode) for it in val_real]

    # Mask bank & rect sizes
    bank, rect_sizes = [], []
    for it in tr_all:
        if it["method"] == a.method and it["image_name"] in val_names:
            continue
        m = it["mask"]
        if is_rect(m):
            if it["method"] == a.method:
                ys, xs = np.nonzero(m)
                rect_sizes.append((int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        else:
            bank.append(m)
    bank = np.stack(bank)

    # Pure clean COCO covers!
    files = sorted((ROOT / "coco_dl" / "val2017").glob("*.jpg"))
    rng.shuffle(files)
    covers = []
    for f in files[:a.n_coco]:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        m = min(w, h)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((512, 512), Image.BICUBIC)
        covers.append(np.asarray(im, dtype=np.uint8))
    covers = np.stack(covers)

    frame_pad = 16 if a.method == "stegastamp" else 0
    wm = load_watermark(a.method, device="cuda")
    t_e = time.time()
    W = np.stack([embed_full(wm, a.method, c, rng, frame_pad) for c in covers])
    RY, RM = [], []
    for c in covers[:min(len(covers), 800)]:
        bw, bh = rect_sizes[rng.integers(len(rect_sizes))]
        x0, y0 = int(rng.integers(0, 513 - bw)), int(rng.integers(0, 513 - bh))
        try:
            RY.append(embed_rect(wm, c, (x0, y0, x0 + bw, y0 + bh), rng))
            m = np.zeros((512, 512), np.uint8)
            m[y0:y0 + bh, x0:x0 + bw] = 1
            RM.append(m)
        except Exception:
            continue
    RY, RM = np.stack(RY), np.stack(RM)
    print(f"[{a.tag}] embedded {len(W)} clean COCO covers + {len(RY)} rects in {time.time() - t_e:.1f}s", flush=True)

    # Keep pools on GPU (uint8, ~2.5 GB total)
    I_g = torch.from_numpy(np.ascontiguousarray(covers)).cuda()
    W_g = torch.from_numpy(np.ascontiguousarray(W)).cuda()
    RY_g = torch.from_numpy(np.ascontiguousarray(RY)).cuda()
    RM_g = torch.from_numpy(np.ascontiguousarray(RM)).cuda()
    realY_g = torch.from_numpy(np.stack([it["image"] for it in train_real])).cuda()
    realM_g = torch.from_numpy(np.stack([it["mask"] for it in train_real])).cuda()
    realMF_g = torch.from_numpy(np.stack([mode_feat_of(mode_json["train"][it["image_name"]], thr_mode)
                                          for it in train_real])).cuda()
    bank_g = torch.from_numpy(np.ascontiguousarray(bank)).cuda()

    base = UNet(32, "band").cuda()
    base.load_state_dict(torch.load(a.init, weights_only=True))
    model = ModeConditionedUNet(base).cuda()

    # Check step-0 score!
    rots_eval = 1 if a.method == "trustmark" else 4
    pv0 = predict_tta(model, val_imgs, val_mf, bs=8, rots=1)
    thr0, m0, bg0, _ = evaluate(pv0, val_gts, val_groups)
    print(f"[{a.tag}] step-0 val (flip-TTA): {m0:.4f} @{thr0} {bg0}", flush=True)
    best = dict(miou=m0, thr=thr0, by_group=bg0, it=0)
    torch.save(model.state_dict(), out / "model.pt")

    opt = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": a.lr, "weight_decay": 1e-4},
        {"params": list(model.m4.parameters()) + list(model.m2.parameters()), "lr": a.lr * 3.0, "weight_decay": 1e-4},
    ])

    budget = a.minutes * 60.0
    t0 = time.time()
    last_eval = t0
    it_n = 0
    loss_ema = None
    refresh_n = 0

    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0:
            break
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * frac)) if frac > 0.05 else max(0.1, frac / 0.05)
        opt.param_groups[0]["lr"] = a.lr * lr_mult
        opt.param_groups[1]["lr"] = a.lr * 3.0 * lr_mult

        # Build full-512 batch: 50% whole synth, 20% rect synth, 30% real
        xs, ms, mfs = [], [], []
        kinds = rng.choice(3, a.bs, p=[0.5, 0.2, 0.3])
        for k in kinds:
            if k == 0:
                idx = int(rng.integers(len(I_g)))
                mk = bank_g[int(rng.integers(len(bank_g)))]
                if rng.random() < 0.5: mk = mk.flip(0)
                if rng.random() < 0.5: mk = mk.flip(1)
                if rng.random() < 0.5: mk = mk.t()
                y = torch.where(mk[..., None].bool(), W_g[idx], I_g[idx])
                area = float(mk.float().mean().item())
                # Realistic noisy synthetic mode score matching real distribution
                if area > 0.5:
                    sc = float(np.clip(rng.normal(0.92, 0.06), thr_mode + 0.05, 1.0))
                else:
                    sc = float(np.clip(rng.normal(0.05, 0.08), -0.1, thr_mode - 0.05))
                # Occasional 2% mode-error simulation so model doesn't blindly trust mode!
                if rng.random() < 0.02:
                    sc = 1.0 - sc
                mf = torch.tensor([sc, float(sc >= thr_mode)], device="cuda", dtype=torch.float32)
            elif k == 1:
                idx = int(rng.integers(len(RY_g)))
                y, mk = RY_g[idx], RM_g[idx]
                sc = float(np.clip(rng.normal(0.03, 0.06), -0.1, thr_mode - 0.05))
                if rng.random() < 0.02:
                    sc = 0.85
                mf = torch.tensor([sc, float(sc >= thr_mode)], device="cuda", dtype=torch.float32)
            else:
                idx = int(rng.integers(len(realY_g)))
                y, mk, mf = realY_g[idx], realM_g[idx], realMF_g[idx]
                if rng.random() < 0.5:
                    y = y.flip(1); mk = mk.flip(1)
            xs.append(y); ms.append(mk); mfs.append(mf)

        x = torch.stack(xs).permute(0, 3, 1, 2).float() / 255.0
        t = F.avg_pool2d(torch.stack(ms)[:, None].float(), 4)
        mf = torch.stack(mfs)

        logits = model(x, mf)
        loss = 0.4 * F.binary_cross_entropy_with_logits(logits, t) + 0.6 * sym_soft_iou_loss(logits, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        loss_ema = loss.item() if loss_ema is None else 0.97 * loss_ema + 0.03 * loss.item()
        it_n += 1

        # Refresh 2 covers every 15 iters
        if it_n % 15 == 0:
            for _ in range(2):
                i = int(rng.integers(len(I_g)))
                c = np.rot90(covers[int(rng.integers(len(covers)))], int(rng.integers(4)))
                if rng.random() < 0.5: c = c[:, ::-1]
                c = np.ascontiguousarray(c)
                w = embed_full(wm, a.method, c, rng, frame_pad)
                I_g[i] = torch.from_numpy(c).cuda()
                W_g[i] = torch.from_numpy(np.ascontiguousarray(w)).cuda()
                refresh_n += 1

        if time.time() - last_eval > a.eval_every * 60.0 or (time.time() - t0) >= budget:
            last_eval = time.time()
            pv = predict_tta(model, val_imgs, val_mf, bs=8, rots=1)
            thr, miou, by_group, _ = evaluate(pv, val_gts, val_groups)
            gtxt = " ".join(f"{k} {v:.3f}" for k, v in by_group.items())
            print(f"[{a.tag}] it {it_n} ({(time.time() - t0)/60:.1f}m) fresh {refresh_n} loss {loss_ema:.4f} | "
                  f"val {miou:.4f} @{thr} ({gtxt})", flush=True)
            if miou > best["miou"]:
                best = dict(miou=miou, thr=thr, by_group=by_group, it=it_n)
                torch.save(model.state_dict(), out / "model.pt")

    torch.save(model.state_dict(), out / "last.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))

    pv = predict_tta(model, val_imgs, val_mf, bs=8, rots=rots_eval)
    thr, miou, by_group, sweep = evaluate(pv, val_gts, val_groups)
    print(f"[{a.tag}] FINAL TTA(rots={rots_eval}) val {miou:.4f} @{thr} {by_group}", flush=True)

    te = load_items(ROOT / "release", "test", [a.method])
    te_mf = [mode_feat_of(mode_json["test"][it["image_name"]], thr_mode) for it in te]
    pt = predict_tta(model, [it["image"] for it in te], te_mf, bs=8, rots=rots_eval)

    np.savez_compressed(out / "pred_val.npz", names=np.array([it["image_name"] for it in val_real]),
                        probs=(pv * 255.0).round().astype(np.uint8))
    np.savez_compressed(out / "pred_test.npz", names=np.array([it["image_name"] for it in te]),
                        probs=(pt * 255.0).round().astype(np.uint8))
    oracle = float(np.mean([miou_image(np.full((128, 128), int(it["full_watermark"]), np.uint8), it["mask128"])
                            for it in val_real]))
    (out / "metrics.json").write_text(json.dumps(dict(
        method=a.method, tag=a.tag, val_miou=float(miou), thr=float(thr), by_group=by_group,
        thr_sweep=sweep, perfect_mode_val=oracle, iterations=it_n, best_it=best.get("it"),
        val_names=sorted(val_names), minutes_total=(time.time() - t_start) / 60.0), indent=1))
    print(f"[{a.tag}] DONE in {(time.time() - t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
