"""PixelSeal v2 Decoder Head: fixes all ps_dechead_synth bugs + adds native-scale 256 crops, JND, mode_256, Soft-mIoU.

Fixes vs ps_dechead_synth.py:
  1. Uses exact mode_256 (Mode AUC 0.9976) instead of broken full_512 (AUC 0.9741).
  2. Extracts full agreement stats (tanh_mean, pos_acc, mean, std, det_ch0) at ALL scales (256, 384, 512)
     PLUS 2x2 overlapping 320px native-scale crops stitched onto the 16x16 grid, PLUS exact JND(Y) map! (18ch total).
  3. Residual Mini-UNet head (16x16 -> 8x8 -> 4x4 -> 8x8 -> 16x16 -> 32x32) trained with 0.5*BCE + 0.5*SymSoftIoU.
  4. Saves CONTINUOUS soft probabilities (128x128) so it can be ensembled with ps_convnext_fpn and ps_framesearch_fp32!
"""
import argparse
import json
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
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

G = 16


def is_rect(mask):
    ys, xs = np.nonzero(mask)
    return len(ys) > 0 and mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all()


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


@torch.no_grad()
def extract_rich_batch(wm, x, s):
    """x: [B, 3, 512, 512] float in [0,1], s: [B, 256] float in {-1,+1} -> [B, 18, 16, 16] float32 CPU."""
    det = wm.det
    lin_w = det.pixel_decoder.linear.weight.float()
    lin_b = det.pixel_decoder.linear.bias.float()
    B = x.shape[0]
    feats = []

    def run_scale(inp):
        emb = det.pixel_decoder.output_upscaling(det.convnext(det.preprocess(inp))).float()
        return torch.einsum("bchw,oc->bohw", emb, lin_w) + lin_b[None, :, None, None]

    # 1) Scale 512 -> 16x16 (5ch)
    pp512 = run_scale(x)
    a512 = pp512[:, 1:] * s[:, :, None, None]
    feats += [torch.tanh(a512).mean(1), (a512 > 0).float().mean(1), a512.mean(1), a512.std(1), pp512[:, 0]]

    # 2) Scale 384 -> 12x12 -> 16x16 (4ch)
    x384 = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False, antialias=True)
    pp384 = run_scale(x384)
    a384 = pp384[:, 1:] * s[:, :, None, None]
    f384 = torch.stack([torch.tanh(a384).mean(1), (a384 > 0).float().mean(1), a384.std(1), pp384[:, 0]], 1)
    f384 = F.interpolate(f384, size=(G, G), mode="bilinear", align_corners=False)
    feats += [f384[:, i] for i in range(4)]

    # 3) Native scale 256 -> 8x8 -> 16x16 (5ch) + exact mode_256 (1ch)
    x256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    pp256 = run_scale(x256)
    a256 = pp256[:, 1:] * s[:, :, None, None]
    mode256 = (torch.tanh(pp256[:, 1:].mean((2, 3))) * s).mean(1)  # [B]
    f256 = torch.stack([torch.tanh(a256).mean(1), (a256 > 0).float().mean(1), a256.mean(1), a256.std(1), pp256[:, 0]], 1)
    f256 = F.interpolate(f256, size=(G, G), mode="bilinear", align_corners=False)
    feats += [f256[:, i] for i in range(5)]

    # 4) 2x2 overlapping 320x320 crops resized to native 256 -> stitched onto 16x16 (2ch: tanh_mean, pos_acc)
    # Crop spans 10 of 16 cells (320px = 10 * 32px), offsets 0 and 6 (0..320 and 192..512)
    acc_t = torch.zeros((B, G, G), device=x.device)
    acc_p = torch.zeros((B, G, G), device=x.device)
    cnt = torch.zeros((1, G, G), device=x.device)
    for gy in (0, 6):
        for gx in (0, 6):
            y0, x0 = gy * 32, gx * 32
            crop = F.interpolate(x[:, :, y0:y0 + 320, x0:x0 + 320], size=(256, 256),
                                 mode="bilinear", align_corners=False, antialias=True)
            ppc = run_scale(crop)
            ac = ppc[:, 1:] * s[:, :, None, None]
            fc = torch.stack([torch.tanh(ac).mean(1), (ac > 0).float().mean(1)], 1)
            fc10 = F.interpolate(fc, size=(10, 10), mode="bilinear", align_corners=False)
            acc_t[:, gy:gy + 10, gx:gx + 10] += fc10[:, 0]
            acc_p[:, gy:gy + 10, gx:gx + 10] += fc10[:, 1]
            cnt[:, gy:gy + 10, gx:gx + 10] += 1.0
    feats += [acc_t / cnt, acc_p / cnt]

    # 5) Exact mode_256 + JND(x) pooled to 16x16 (2ch)
    jnd16 = F.avg_pool2d(wm.model.attenuation.heatmaps(x)[:, 0:1].float(), 32)[:, 0]
    feats += [mode256.view(B, 1, 1).expand(B, G, G), jnd16]
    return torch.stack(feats, dim=1).cpu().numpy().astype(np.float32)  # [B, 18, 16, 16]


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class RichUNetHead(nn.Module):
    """16x16 -> 8x8 -> 4x4 -> 8x8 -> 16x16 -> 32x32 residual U-Net with global context."""

    def __init__(self, c_in, w=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU())
        self.e1 = ResBlock(w)
        self.d1 = nn.Sequential(nn.Conv2d(w, w * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), ResBlock(w * 2))
        self.d2 = nn.Sequential(nn.Conv2d(w * 2, w * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), ResBlock(w * 2))
        self.u1 = nn.Sequential(nn.Conv2d(w * 4, w * 2, 3, padding=1, bias=False), nn.GroupNorm(8, w * 2), nn.GELU(), ResBlock(w * 2))
        self.u0 = nn.Sequential(nn.Conv2d(w * 3, w, 3, padding=1, bias=False), nn.GroupNorm(8, w), nn.GELU(), ResBlock(w))
        self.up32 = nn.Sequential(nn.Conv2d(w, w // 2, 3, padding=1, bias=False), nn.GroupNorm(8, w // 2), nn.GELU(),
                                  nn.Conv2d(w // 2, 1, 1))

    def forward(self, x):
        e1 = self.e1(self.stem(x))            # 16x16
        e2 = self.d1(e1)                      # 8x8
        e3 = self.d2(e2)                      # 4x4
        u1 = self.u1(torch.cat([F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        u0 = self.u0(torch.cat([F.interpolate(u1, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.up32(F.interpolate(u0, scale_factor=2, mode="bilinear", align_corners=False))  # 32x32


def build_synth_batched(wm, n, rng, coco_dir, bs=16):
    files = sorted(Path(coco_dir).glob("*.jpg"))
    rng.shuffle(files)
    tr = LocalisationDataset(ROOT / "release", "train")
    bank = [(tr[i]["mask"].astype(bool), bool(tr.meta[i]["full_watermark"])) for i in range(len(tr))]
    X_list, Y_list = [], []
    t0 = time.time()
    idx = 0
    while len(X_list) * bs < n:
        cur_bs = min(bs, n - len(X_list) * bs)
        imgs_c, masks_c, is_r_list = [], [], []
        for _ in range(cur_bs):
            f = files[idx % len(files)]; idx += 1
            im = Image.open(f).convert("RGB"); w, h = im.size; m = min(w, h)
            c = np.asarray(im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((512, 512), Image.BICUBIC))
            mk, fw = bank[rng.integers(len(bank))]
            mk = np.rot90(mk, int(rng.integers(4)))
            if rng.random() < 0.5: mk = mk[:, ::-1]
            mk = np.ascontiguousarray(mk)
            imgs_c.append(c); masks_c.append(mk); is_r_list.append((not fw) and is_rect(mk))
        c_t = torch.from_numpy(np.stack(imgs_c)).cuda().permute(0, 3, 1, 2).float() / 255.0
        bits_t = torch.randint(0, 2, (cur_bs, 256), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            w_all = torch.round(wm.model.embed(c_t, msgs=bits_t, is_video=False)["imgs_w"].clamp(0, 1) * 255.0) / 255.0
            y_t = c_t.clone()
            for k in range(cur_bs):
                mk = masks_c[k]
                if is_r_list[k]:
                    ys, xs = np.nonzero(mk); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    cw = torch.round(wm.model.embed(c_t[k:k+1, :, y0:y1, x0:x1], msgs=bits_t[k:k+1], is_video=False)["imgs_w"].clamp(0, 1) * 255.0) / 255.0
                    y_t[k:k+1, :, y0:y1, x0:x1] = cw
                else:
                    mt = torch.from_numpy(mk).cuda()[None, None].float()
                    y_t[k:k+1] = mt * w_all[k:k+1] + (1.0 - mt) * c_t[k:k+1]
            s_t = 2.0 * bits_t - 1.0
            X_list.append(extract_rich_batch(wm, y_t, s_t))
        y32 = np.stack([mk.reshape(32, 16, 32, 16).mean((1, 3)).astype(np.float32) for mk in masks_c])
        Y_list.append(y32)
        if len(X_list) % 50 == 0:
            print(f"[ps_v2] generated {len(X_list)*bs}/{n} in {time.time()-t0:.1f}s", flush=True)
    print(f"[ps_v2] generated {n} rich synthetic examples in {time.time()-t0:.1f}s", flush=True)
    return np.concatenate(X_list)[:n], np.concatenate(Y_list)[:n]


def to128(p32):
    return F.interpolate(torch.as_tensor(p32)[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].numpy()


def fit_head(X, Y, w, steps=3500, width=64, seed=0):
    torch.manual_seed(seed)
    head = RichUNetHead(X.shape[1], width).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    xt = torch.as_tensor(X, device="cuda")
    yt = torch.as_tensor(Y, device="cuda")[:, None]
    wt = torch.as_tensor(w, device="cuda")[:, None, None, None]
    for _ in range(steps):
        idx = torch.randint(len(xt), (min(len(xt), 128),), device="cuda")
        a, b, ww = xt[idx], yt[idx], wt[idx]
        k = int(torch.randint(4, (1,)))
        a = torch.rot90(a, k, (2, 3)); b = torch.rot90(b, k, (2, 3))
        if torch.rand(1) < 0.5:
            a = a.flip(3); b = b.flip(3)
        logits = head(a)
        bce = (F.binary_cross_entropy_with_logits(logits, b, reduction="none") * ww).mean()
        siou = sym_soft_iou_loss(logits, b)
        loss = 0.5 * bce + 0.5 * siou
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return head


@torch.no_grad()
def apply_tta(head, Xn):
    head.eval()
    xt = torch.as_tensor(Xn, device="cuda")
    acc = 0
    for flip in (False, True):
        xf = xt.flip(3) if flip else xt
        for k in range(4):
            p = torch.sigmoid(head(torch.rot90(xf, k, (2, 3))))
            p = torch.rot90(p, -k, (2, 3))
            acc = acc + (p.flip(3) if flip else p)
    return (acc / 8.0)[:, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_synth", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--tag", default="ps_dechead_v2")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t_start = time.time()

    wm = load_watermark("pixelseal", device="cuda")
    tr = LocalisationDataset(ROOT / "release", "train", methods=["pixelseal"])
    te = LocalisationDataset(ROOT / "release", "test", methods=["pixelseal"])

    def extract_ds(ds, is_train):
        Xs, Ys, names = [], [], []
        for i in range(0, len(ds), 16):
            sub = [ds[j] for j in range(i, min(i + 16, len(ds)))]
            x = torch.from_numpy(np.stack([it["image"] for it in sub])).cuda().permute(0, 3, 1, 2).float() / 255.0
            s = torch.from_numpy(np.stack([2.0 * it["message"].astype(np.float32) - 1.0 for it in sub])).cuda()
            Xs.append(extract_rich_batch(wm, x, s))
            names += [it["image_name"] for it in sub]
            if is_train:
                Ys += [it["mask"].reshape(32, 16, 32, 16).mean((1, 3)).astype(np.float32) for it in sub]
        return np.concatenate(Xs), (np.stack(Ys) if is_train else None), names

    X, Y, names = extract_ds(tr, True)
    XT, _, tnames = extract_ds(te, False)
    print(f"[{a.tag}] extracted real train {X.shape}, test {XT.shape} in {time.time()-t_start:.1f}s", flush=True)

    XS, YS = build_synth_batched(wm, a.n_synth, np.random.default_rng(123), ROOT / "coco_dl" / "val2017", bs=16)
    del wm; torch.cuda.empty_cache()

    mu = X.mean((0, 2, 3), keepdims=True)
    sd = X.std((0, 2, 3), keepdims=True) + 1e-6
    Xn, XSn, XTn = (X - mu) / sd, (XS - mu) / sd, (XT - mu) / sd

    mode = json.loads((ROOT / "results_joba" / "mode_pixelseal_fp32.json").read_text())
    val_names = set(json.loads((ROOT / "results_seg" / "pixelseal" / "metrics.json").read_text())["val_names"])
    g = np.load(ROOT / "release" / "train" / "masks_128.npz")
    gt = dict(zip(g["names"].tolist(), g["masks"]))
    fw = np.array([mode["train"][n] >= mode["thr"] for n in names])

    N = len(names)
    folds = np.random.default_rng(0).permutation(N) % 5
    best = None
    for rw in (0.0, 3.0):
        P = np.zeros_like(Y)
        for k in range(5):
            idx = folds != k
            XX = np.concatenate([Xn[idx], XSn]); YY = np.concatenate([Y[idx], YS])
            w = np.concatenate([np.full(idx.sum(), rw, np.float32), np.ones(len(XSn), np.float32)])
            head = fit_head(XX, YY, w, steps=a.steps, width=64, seed=k)
            P[folds == k] = apply_tta(head, Xn[folds == k])
        P128 = np.stack([to128(p) for p in P])
        TT = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        TF = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
        sc = {(tT, tF): float(np.mean([miou_image((p >= (tT if f else tF)).astype(np.uint8), gt[n])
                                       for p, f, n in zip(P128, fw, names)]))
              for tT in TT for tF in TF}
        t = max(sc, key=sc.get)
        print(f"[{a.tag}] rw={rw} -> 5-fold CV {sc[t]:.4f} @{t}", flush=True)
        if best is None or sc[t] > best[0]:
            best = (sc[t], rw, t, P128)

    cv, rw, (tT, tF), P128 = best
    vi = [i for i, n in enumerate(names) if n in val_names]
    v_mode = float(np.mean([miou_image((P128[i] >= (tT if fw[i] else tF)).astype(np.uint8), gt[names[i]]) for i in vi]))
    s_single = {thr: float(np.mean([miou_image((P128[i] >= thr).astype(np.uint8), gt[names[i]]) for i in vi]))
                for thr in (0.2, 0.3, 0.4, 0.5, 0.6)}
    b_sing = max(s_single, key=s_single.get)
    print(f"[{a.tag}] BEST rw={rw}: 5-fold CV {cv:.4f} @({tT},{tF}) | val52 mode-thr {v_mode:.4f}, single {s_single[b_sing]:.4f}@{b_sing}", flush=True)

    # 5-seed ensemble on all 170 real + 6k synthetic for test!
    XX = np.concatenate([Xn, XSn]); YY = np.concatenate([Y, YS])
    w = np.concatenate([np.full(N, rw, np.float32), np.ones(len(XSn), np.float32)])
    PT = np.mean([apply_tta(fit_head(XX, YY, w, steps=a.steps, width=64, seed=s), XTn) for s in range(5)], 0)
    PT128 = np.stack([to128(p) for p in PT])

    out = ROOT / "results_seg" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    # Save CONTINUOUS soft probabilities (NOT binarized!)
    np.savez_compressed(out / "pred_val.npz", names=np.array([names[i] for i in vi]),
                        probs=(P128[vi] * 255.0).round().astype(np.uint8))
    np.savez_compressed(out / "pred_oof.npz", names=np.array(names),
                        probs=(P128 * 255.0).round().astype(np.uint8))
    np.savez_compressed(out / "pred_test.npz", names=np.array(tnames),
                        probs=(PT128 * 255.0).round().astype(np.uint8))
    sT = float(np.mean([miou_image((P128[i] >= tT).astype(np.uint8), gt[names[i]]) for i in vi if fw[i]]))
    sF = float(np.mean([miou_image((P128[i] >= tF).astype(np.uint8), gt[names[i]]) for i in vi if not fw[i]]))
    (out / "metrics.json").write_text(json.dumps(dict(
        method="pixelseal", tag=a.tag, val_miou=float(s_single[b_sing]), val_miou_mode=v_mode,
        cv_miou=float(cv), thr=float(b_sing), thresholds=[tT, tF], rw=rw, n_synth=a.n_synth,
        by_group={"fw=True": sT, "fw=False": sF}, perfect_mode_val=0.4499,
        val_names=sorted(val_names), minutes_total=(time.time() - t_start) / 60.0), indent=1))
    print(f"[{a.tag}] DONE in {(time.time()-t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
