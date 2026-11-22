"""Learned watermark segmenter, one model per method (training-free decoder probing tops out at ~0.5).

Training data, built with the dataset's own embedders (wmloc.watermarks):
  (a) whole-image composites: cover I, W = embed(I, random msg), Y = M*W + (1-M)*I with M a real train mask shape
      (fw=True holes and fw=False object masks, randomly flipped/transposed). Covers are the images of the OTHER two
      methods (train + test): they carry no watermark of this method, so no download is needed. StegaStamp is embedded
      on a frame padded by 0-16 px per side and cropped back (hack_test: the real frame is ~544 px, varying).
  (b) rectangle composites: the watermark is embedded on the crop only (outpainting: decoding the crop gives 1.0),
      with real rectangle sizes of this method at random positions.
  (c) real train images of this method (70%; the other 30% are the validation set).
Model: small U-Net on full-resolution 256 px crops (no resizing: it would blur the residual), output on the official
4x4 grid (128 x 128 for a 512 image). fp32/TF32 only (fp16 gave NaNs for PixelSeal). Time-based cosine schedule;
the checkpoint with the best validation mIoU (threshold tuned on validation) is used for the test predictions.

    python experiments/seg_train.py --method trustmark --minutes 55 --out results_seg
Writes <out>/<tag>/{model.pt, metrics.json, pred_test.npz, pred_val.npz, log via stdout}.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import downsample_mask as downsample, miou_image  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


# ------------------------------------------------------------------------------------------------ data
def is_rect(mask):
    ys, xs = np.nonzero(mask)
    return len(ys) > 0 and mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all()


def load_items(data, split, methods):
    ds = LocalisationDataset(data, split, methods=list(methods))
    return [ds[i] for i in range(len(ds))]


def embed_full(wm, method, img, rng, frame_pad):
    """Whole-image watermark. frame_pad > 0: embed on a frame reflect-padded by 0..frame_pad px per side, then crop
    back (hack_test: the real StegaStamp frame is ~544 px with a varying offset; probably true for all methods)."""
    msg = rng.integers(0, 2, wm.nbits)
    if frame_pad > 0:
        l, t, r, b = (int(v) for v in rng.integers(0, frame_pad + 1, 4))
        padded = np.pad(img, ((t, b), (l, r), (0, 0)), mode="reflect")
        return wm.embed(padded, msg)[t:t + 512, l:l + 512]
    return wm.embed(img, msg)


def embed_rect(wm, img, box, rng):
    x0, y0, x1, y1 = box
    out = img.copy()
    out[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(img[y0:y1, x0:x1]), rng.integers(0, 2, wm.nbits))
    return out


def build_synthetic(method, covers, rect_sizes, rng, cache, frame_pad):
    """Returns W (N x 512 x 512 x 3 uint8: covers fully watermarked) and rect composites (Y, M)."""
    if cache.exists():
        z = np.load(cache)
        print(f"loaded synthetic data from {cache}", flush=True)
        return z["W"], z["RY"], z["RM"]
    wm = load_watermark(method, device="cuda")
    t0 = time.time()
    W = np.stack([embed_full(wm, method, c, rng, frame_pad) for c in covers])
    print(f"embedded {len(W)} whole covers in {time.time() - t0:.0f}s", flush=True)
    RY, RM = [], []
    for c in covers:
        w, h = rect_sizes[rng.integers(len(rect_sizes))]
        x0, y0 = int(rng.integers(0, 513 - w)), int(rng.integers(0, 513 - h))
        try:
            RY.append(embed_rect(wm, c, (x0, y0, x0 + w, y0 + h), rng))
        except Exception as e:  # tiny or odd crops: skip
            print("rect embed failed:", w, h, e)
            continue
        m = np.zeros((512, 512), np.uint8); m[y0:y0 + h, x0:x0 + w] = 1; RM.append(m)
    RY, RM = np.stack(RY), np.stack(RM)
    print(f"embedded {len(RY)} rectangle composites in {time.time() - t0:.0f}s", flush=True)
    np.savez(cache, W=W, RY=RY, RM=RM)
    del wm
    torch.cuda.empty_cache()
    return W, RY, RM


def d4(img, rng):
    """Random flip / 90-degree rotation of a COVER (before embedding, so the watermark pattern stays natural)."""
    img = np.rot90(img, int(rng.integers(4)))
    return np.ascontiguousarray(img[:, ::-1] if rng.random() < 0.5 else img)


class Refresher:
    """Fresh data on the fly (E13: with one fixed watermarked twin per cover, weak watermarks are memorised per cover
    instead of learned). Every call re-embeds k training covers (random flip/rotation, new message) and k rectangle
    composites, writing them over random slots of the sampler's GPU pools."""

    def __init__(self, method, covers, rect_sizes, frame_pad, sampler, seed):
        self.wm = load_watermark(method, device="cuda")
        self.method, self.covers, self.rect_sizes, self.frame_pad, self.s = method, covers, rect_sizes, frame_pad, sampler
        self.rng = np.random.default_rng(seed + 1000)
        self.n_done = 0

    def __call__(self, k):
        dev = self.s.I.device
        for _ in range(k):
            i = int(self.rng.integers(len(self.s.I)))
            c = d4(self.covers[self.rng.integers(len(self.covers))], self.rng)
            w = embed_full(self.wm, self.method, c, self.rng, self.frame_pad)
            self.s.I[i] = torch.from_numpy(c).to(dev); self.s.W[i] = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
            j = int(self.rng.integers(len(self.s.RY)))
            c = d4(self.covers[self.rng.integers(len(self.covers))], self.rng)
            bw, bh = self.rect_sizes[self.rng.integers(len(self.rect_sizes))]
            x0, y0 = int(self.rng.integers(0, 513 - bw)), int(self.rng.integers(0, 513 - bh))
            try:
                y = embed_rect(self.wm, c, (x0, y0, x0 + bw, y0 + bh), self.rng)
            except Exception:
                continue
            m = np.zeros((512, 512), np.uint8); m[y0:y0 + bh, x0:x0 + bw] = 1
            self.s.RY[j] = torch.from_numpy(y).to(dev); self.s.RM[j] = torch.from_numpy(m).to(dev)
            self.n_done += 1


class Sampler:
    """Builds training batches on the GPU from uint8 tensors."""

    def __init__(self, I, W, RY, RM, realY, realM, bank, probs, dev, seed):
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
        self.I, self.W, self.RY, self.RM = g(I), g(W), g(RY), g(RM)
        self.realY, self.realM, self.bank = g(realY), g(realM), g(bank)
        self.probs = probs
        self.rng = np.random.default_rng(seed)

    def batch(self, B, crop):
        xs, ms = [], []
        kinds = self.rng.choice(3, B, p=self.probs)
        for k in kinds:
            if k == 0:
                i = self.rng.integers(len(self.I))
                m = self.bank[self.rng.integers(len(self.bank))]
                if self.rng.random() < 0.5:
                    m = m.flip(0)
                if self.rng.random() < 0.5:
                    m = m.flip(1)
                if self.rng.random() < 0.5:
                    m = m.t()
                y = torch.where(m[..., None].bool(), self.W[i], self.I[i])
            elif k == 1:
                j = self.rng.integers(len(self.RY)); y, m = self.RY[j], self.RM[j]
            else:
                j = self.rng.integers(len(self.realY)); y, m = self.realY[j], self.realM[j]
            cx, cy = (int(v) * 4 for v in self.rng.integers(0, (512 - crop) // 4 + 1, 2))  # keep the 4x4 grid
            xs.append(y[cy:cy + crop, cx:cx + crop]); ms.append(m[cy:cy + crop, cx:cx + crop])
        x = torch.stack(xs).permute(0, 3, 1, 2).float() / 255
        t = F.avg_pool2d(torch.stack(ms)[:, None].float(), 4)          # soft target on the 4x4 grid
        return x, t


# ----------------------------------------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def gauss_kernel(sigma):
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    return k / k.sum()


class BandPass(nn.Module):
    """Fixed difference-of-Gaussians pyramid: x - G1 x, G1 x - G2 x, G2 x - G4 x, G4 x - G8 x (per colour channel),
    each scaled so natural-image content has roughly unit spread. Measured residual-to-content ratios: TrustMark peaks
    in the 2-8 px bands, PixelSeal in the finest band; a 3x3 high-pass alone mostly sees the weakest part."""
    SIGMAS = (1, 2, 4, 8)

    def __init__(self, gain=25.0):
        super().__init__()
        self.gain = gain
        for s in self.SIGMAS:
            self.register_buffer(f"k{s}", gauss_kernel(s))

    def blur(self, x, s):
        k = getattr(self, f"k{s}"); r = len(k) // 2; c = x.shape[1]
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)

    def forward(self, x):
        out, prev = [], x
        for s in self.SIGMAS:
            b = self.blur(x, s); out.append((prev - b) * self.gain); prev = b
        return torch.cat(out, 1)


class UNet(nn.Module):
    """Full-resolution stem, encoder to /16, decoder back to /4 (the official 128 x 128 grid for 512 input).
    inp="hp3": image + 3x3 high-pass (6 channels); inp="band": image + 4 difference-of-Gaussian bands (15 channels)."""

    def __init__(self, w=32, inp="hp3"):
        super().__init__()
        self.inp = inp
        # "luma": PixelSeal adds the same value to R, G and B, so band-pass the channel mean (colour content averages
        # out, the watermark does not) with a higher gain
        self.band = BandPass() if inp == "band" else (BandPass(gain=40.0) if inp == "luma" else None)
        c = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        n_in = {"band": 15, "luma": 7, "hp3": 6}[inp]
        self.e0, self.e1, self.e2 = block(n_in, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.e3, self.e4 = block(c[2], c[3]), block(c[3], c[4])
        self.d3, self.d2 = block(c[4] + c[3], c[3]), block(c[3] + c[2], c[2])
        self.head = nn.Conv2d(c[2], 1, 1)

    def forward(self, x):
        if self.inp == "luma":
            x = torch.cat([(x - 0.5) * 4, self.band(x.mean(1, keepdim=True))], 1)
        elif self.band is not None:
            x = torch.cat([(x - 0.5) * 4, self.band(x)], 1)
        else:
            hp = x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)   # fine residual, amplified
            x = torch.cat([(x - 0.5) * 4, hp * 20], 1)
        e0 = self.e0(x)
        e1 = self.e1(F.max_pool2d(e0, 2)); e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2)); e4 = self.e4(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        d3 = self.d3(torch.cat([up(e4), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        return self.head(d2)


@torch.no_grad()
def predict(model, images, dev, bs=8):
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255
        out.append(torch.sigmoid(model(x))[:, 0].cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate(probs, gts, groups):
    res = {}
    for thr in THRESHOLDS:
        s = [miou_image((p >= thr).astype(np.uint8), g) for p, g in zip(probs, gts)]
        res[thr] = (float(np.mean(s)), {k: float(np.mean([v for v, gg in zip(s, groups) if gg == k]))
                                         for k in sorted(set(groups))})
    best = max(res, key=lambda t: res[t][0])
    return best, res[best][0], res[best][1], {t: r[0] for t, r in res.items()}


def group_of(it):
    return "fw=True" if it["full_watermark"] else ("rect" if is_rect(it["mask"]) else "object")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--data", default="release"); ap.add_argument("--out", default="results_seg")
    ap.add_argument("--tag", default=None, help="output subfolder (default: method)")
    ap.add_argument("--minutes", type=float, default=55); ap.add_argument("--eval_every", type=float, default=3,
                                                                          help="minutes between validations")
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probs", type=float, nargs=3, default=[0.5, 0.2, 0.3], help="mix of (a) whole, (b) rect, (c) real")
    ap.add_argument("--extra_covers", default=None, help="folder of clean .jpg images added to the cover pool")
    ap.add_argument("--n_extra", type=int, default=3000, help="max external covers")
    ap.add_argument("--n_covers", type=int, default=0, help="0 = all covers"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--inp", choices=["hp3", "band", "luma"], default="hp3", help="input filters (see UNet)")
    ap.add_argument("--init", default=None, help="start from these weights (same --width / --inp)")
    ap.add_argument("--refresh_k", type=int, default=0,
                    help="fresh embeddings (cover + rectangle) per refresh; 0 = fixed pools as in the first run")
    ap.add_argument("--refresh_every", type=int, default=10, help="iterations between refreshes")
    ap.add_argument("--frame_pad", type=int, default=-1,
                    help="max extra px per side of the embedding frame (-1: 16 for stegastamp, 0 otherwise)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    tag = a.tag or a.method
    out = Path(a.out) / tag; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    t_start = time.time()

    # real data of this method, stratified 70/30 split (fixed seed 0 so variants share the validation set)
    tr_all = load_items(a.data, "train", METHODS)
    real = [it for it in tr_all if it["method"] == a.method]
    split_rng = np.random.default_rng(0)
    val_names = set()
    for fw in (True, False):
        names = sorted(it["image_name"] for it in real if bool(it["full_watermark"]) == fw)
        split_rng.shuffle(names); val_names |= set(names[:int(round(a.val_frac * len(names)))])
    train_real = [it for it in real if it["image_name"] not in val_names]
    val_real = [it for it in real if it["image_name"] in val_names]

    # mask bank (shapes of holes and objects; excludes this method's validation images) and rectangle sizes
    bank, rect_sizes = [], []
    for it in tr_all:
        if it["method"] == a.method and it["image_name"] in val_names:
            continue
        m = it["mask"]
        if is_rect(m):
            if it["method"] == a.method:
                ys, xs = np.nonzero(m); rect_sizes.append((xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        else:
            bank.append(m)
    bank = np.stack(bank)

    # covers: every image (train + test) of the other two methods
    others = [m for m in METHODS if m != a.method]
    covers = [it["image"] for it in tr_all if it["method"] in others]
    covers += [it["image"] for it in load_items(a.data, "test", others)]
    if a.extra_covers:   # clean external images (e.g. COCO), centre square crop -> 512; shuffled, then capped
        files = sorted(Path(a.extra_covers).glob("*.jpg"))
        np.random.default_rng(a.seed).shuffle(files)
        for f in files[:a.n_extra]:
            im = Image.open(f).convert("RGB"); w, h = im.size; m = min(w, h)
            im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((512, 512), Image.BICUBIC)
            covers.append(np.asarray(im))
        print(f"[{tag}] added {min(len(files), a.n_extra)} external covers from {a.extra_covers}", flush=True)
    if a.n_covers:
        covers = covers[:a.n_covers]
    covers = np.stack(covers)
    print(f"[{tag}] real train {len(train_real)} / val {len(val_real)}, covers {len(covers)}, "
          f"mask bank {len(bank)}, rect sizes {len(rect_sizes)}", flush=True)
    frame_pad = a.frame_pad if a.frame_pad >= 0 else (16 if a.method == "stegastamp" else 0)
    suffix = "" if frame_pad == (16 if a.method == "stegastamp" else 0) else f"_p{frame_pad}"   # first run's names
    cache = Path(a.out) / f"synth_{a.method}_n{len(covers)}_s{a.seed}{suffix}.npz"
    W, RY, RM = build_synthetic(a.method, covers, rect_sizes, rng, cache, frame_pad)

    # hold out the last 10% of covers (and their rectangle composites) as a SYNTHETIC validation set: real val low
    # but synthetic val high means a synthetic-vs-real mismatch, both low means the model has not learned enough
    nc, nr = int(0.9 * len(covers)), int(0.9 * len(RY))
    vrng = np.random.default_rng(123)
    syn_imgs, syn_gts = [], []
    for i in range(nc, len(covers)):
        m = bank[vrng.integers(len(bank))].astype(bool)
        syn_imgs.append(np.where(m[..., None], W[i], covers[i])); syn_gts.append(downsample(m))
    for j in range(nr, len(RY)):
        syn_imgs.append(RY[j]); syn_gts.append(downsample(RM[j].astype(bool)))
    covers, W, RY, RM = covers[:nc], W[:nc], RY[:nr], RM[:nr]
    print(f"[{tag}] frame_pad {frame_pad}, input {a.inp}; synthetic val {len(syn_imgs)} composites", flush=True)

    sampler = Sampler(covers, W, RY, RM, np.stack([it["image"] for it in train_real]),
                      np.stack([it["mask"] for it in train_real]), bank, a.probs, dev, a.seed)
    refresher = Refresher(a.method, covers, rect_sizes, frame_pad, sampler, a.seed) if a.refresh_k > 0 else None
    val_imgs = [it["image"] for it in val_real]; val_gts = [it["mask128"] for it in val_real]
    val_groups = [group_of(it) for it in val_real]
    oracle_mode = np.mean([miou_image(np.full((128, 128), int(it["full_watermark"]), np.uint8), it["mask128"])
                           for it in val_real])
    print(f"[{tag}] validation reference: perfect mode (all ones / all zeros) = {oracle_mode:.4f}", flush=True)

    model = UNet(a.width, a.inp).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, weights_only=True))
        print(f"[{tag}] initialised from {a.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    budget = a.minutes * 60
    t0 = time.time(); last_eval = t0; it_n = 0; best = dict(miou=-1.0)
    loss_ema = None
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + math.cos(math.pi * frac)) if frac > 0.03 else frac / 0.03)
        x, t = sampler.batch(a.bs, a.crop)
        loss = F.binary_cross_entropy_with_logits(model(x), t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        it_n += 1
        if refresher is not None and it_n % a.refresh_every == 0:
            refresher(a.refresh_k)
        if time.time() - last_eval > a.eval_every * 60 or (time.time() - t0) >= budget:
            last_eval = time.time()
            thr, miou, by_group, _ = evaluate(predict(model, val_imgs, dev), val_gts, val_groups)
            syn = np.mean([miou_image((p >= 0.5).astype(np.uint8), g)
                           for p, g in zip(predict(model, syn_imgs, dev), syn_gts)])
            gtxt = " ".join(f"{k} {v:.3f}" for k, v in by_group.items()) + f") (synthetic val {syn:.3f}"
            fresh = f" fresh {refresher.n_done}" if refresher is not None else ""
            print(f"[{tag}] it {it_n} ({(time.time() - t0) / 60:.1f} min){fresh} loss {loss_ema:.4f} | val mIoU {miou:.4f} "
                  f"@thr {thr} ({gtxt})", flush=True)
            if miou > best["miou"]:
                best = dict(miou=miou, thr=thr, by_group=by_group, it=it_n)
                torch.save(model.state_dict(), out / "model.pt")
    torch.save(model.state_dict(), out / "last.pt")          # final weights (model.pt = best validation)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    thr, miou, by_group, sweep = evaluate(predict(model, val_imgs, dev), val_gts, val_groups)
    print(f"[{tag}] BEST val mIoU {miou:.4f} @thr {thr} {by_group} (perfect mode {oracle_mode:.4f}); "
          f"{it_n} iterations", flush=True)

    te = load_items(a.data, "test", [a.method])
    p_te = predict(model, [it["image"] for it in te], dev)
    p_val = predict(model, val_imgs, dev)
    np.savez_compressed(out / "pred_test.npz", names=np.array([it["image_name"] for it in te]),
                        probs=(p_te * 255).round().astype(np.uint8))
    np.savez_compressed(out / "pred_val.npz", names=np.array([it["image_name"] for it in val_real]),
                        probs=(p_val * 255).round().astype(np.uint8))
    (out / "metrics.json").write_text(json.dumps(dict(
        method=a.method, tag=tag, val_miou=miou, thr=thr, by_group=by_group, thr_sweep=sweep,
        perfect_mode_val=float(oracle_mode), iterations=it_n, best_it=best.get("it"), args=vars(a),
        val_names=sorted(val_names), minutes_total=(time.time() - t_start) / 60), indent=1))
    print(f"[{tag}] wrote predictions for {len(te)} test images; total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
