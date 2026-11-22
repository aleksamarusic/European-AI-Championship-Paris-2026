"""Stage A + B submission (SOLUTION_DESIGN.md sections 3, 4.1, 4.4, 6):

  1. Mode classifier: full-image decode vs the known message. The threshold is calibrated per method on train.
     Rescue (E1 errors were large fw=False frames): images with an ambiguous full-image score are also
     frame-searched, and a frame of <= 50% area decoding `margin` better than the full image means fw=False.
     The margin is tuned on train; "no rescue" is always a candidate.
  2. Predicted full_watermark=False: frame search. Score many windows by decoding them (batched roi_align
     crops; StegaStamp pre-filtered by local noise contrast), keep diverse top-K, refine each edge by
     coordinate ascent (16 -> 1 px) using the wrapper's exact resize, then (TrustMark/PixelSeal) pull edges
     inwards while the score stays at the peak, because over-large boxes score almost as well (E2).
     Output the best rectangle.
  3. Predicted full_watermark=True: all ones minus an optional hole taken from a presence map (PixelSeal /
     TrustMark: per-position message agreement; StegaStamp: NCC with the re-embedded residual). The hole
     size quantile q is tuned on train, and q = 0 (no hole) is always a candidate, so it cannot do worse on
     train than all ones.

One job per method (they can run in parallel on separate GPUs); test images can be sharded further:
    python experiments/frame_search.py --methods trustmark                  # train eval + test preds
    python experiments/frame_search.py --methods stegastamp --shard 0/2     # half of the test images
    python experiments/frame_search.py --merge --out results                # -> submission.pt

Per-image frame-search results are cached in results/fs_cache_<method>.json, so a timed-out job resumes.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torchvision.ops import roi_align

from common import (METHODS, RES, bbox, best_threshold_acc, device, highpass, img_tensor, load_wm, local_ncc,
                    prep_to_res, sign_vec)
from e3_e4_maps import decoder_maps, embed_f
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image

MAX_AR = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}
# From E2 on train: half-width of the decode peak is ~16 px (TrustMark, PixelSeal) and ~4 px (StegaStamp).
DEFAULT_MAX_STRIDE = {"pixelseal": 16, "trustmark": 16, "stegastamp": 8}
# Windows kept after the noise-contrast pre-filter (0 = decode all). Train recall of a window with IoU >= 0.5 to
# the true rectangle, StegaStamp: top 6k 0.40, top 15k 0.65, all 187k 0.95. Raise it if GPU throughput allows.
DEFAULT_PREFILTER = {"pixelseal": 0, "trustmark": 0, "stegastamp": 20000}
# Score tolerance for pulling edges inwards after the ascent (0 = off). StegaStamp's peak is already sharp.
DEFAULT_SHRINK_TOL = {"pixelseal": 0.01, "trustmark": 0.01, "stegastamp": 0.0}
# Window sizes. PixelSeal rectangles under ~3% of the image don't decode anyway (E2), so skip the tiny sizes.
DEFAULT_SIZES = {"pixelseal": [48, 64, 96, 128, 192, 256, 384, 512],
                 "trustmark": [24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
                 "stegastamp": [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]}
W = 512


# ----------------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def decode_at_res(wm, method, batch, fp16=True):
    """batch: N x 3 x R x R in [0, 1] already at the decoder resolution -> N x nbits logits (CPU)."""
    if method == "stegastamp":
        return wm.logits(batch.float().cpu(), bs=64).float()
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and torch.cuda.is_available()):
        return wm.logits(batch).float().cpu()


def soft_acc(lg, s, mode="tanh"):
    """(score, bit accuracy). mode "tanh": mean tanh(logit)*s, in [-1, 1], saturates at 1 for confident decodes
    (used for the mode threshold and coarse ranking). mode "margin": mean logit*s, does not saturate, so it still
    rewards the exactly aligned box when the tanh score is already ~1 (used for edge refinement)."""
    m = lg * s
    score = torch.tanh(lg) * s if mode == "tanh" else m
    return score.mean(1).numpy(), (m > 0).float().mean(1).numpy()


@torch.no_grad()
def score_windows(wm, method, x_dev, s, boxes, bs, fp16):
    """Fast coarse scoring: crop + resize with roi_align (adaptive sampling ~ antialiasing)."""
    R = RES[method]; out = []
    for i in range(0, len(boxes), bs):
        b = torch.from_numpy(boxes[i:i + bs]).to(x_dev.device)
        rois = torch.cat([torch.zeros(len(b), 1, device=b.device), b], 1)
        crops = roi_align(x_dev, rois, output_size=(R, R), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        out.append(soft_acc(decode_at_res(wm, method, crops.clamp(0, 1), fp16), s)[0])
    return np.concatenate(out)


@torch.no_grad()
def score_exact(wm, method, x_dev, s, boxes, fp16, mode="tanh"):
    """Exact scoring: same resize as the wrapper (antialiased bilinear / bicubic)."""
    crops = torch.cat([prep_to_res(x_dev[..., y0:y1, x0:x1], method) for x0, y0, x1, y1 in boxes])
    return soft_acc(decode_at_res(wm, method, crops, fp16), s, mode)


# ---------------------------------------------------------------------------------------------- frame search
def gen_windows(max_ar, sizes, stride_frac, max_stride=10 ** 9):
    """Stride = min(stride_frac * size, max_stride). E2: the decode peak is about +-16 px wide for TrustMark and
    PixelSeal but only about +-4 px for StegaStamp, in absolute pixels, so large windows need a capped stride."""
    out = []
    for w in sizes:
        for h in sizes:
            if max(w, h) / min(w, h) > max_ar + 1e-6:
                continue
            sx, sy = max(4, min(int(w * stride_frac), max_stride)), max(4, min(int(h * stride_frac), max_stride))
            xs = list(range(0, W - w + 1, sx)); ys = list(range(0, W - h + 1, sy))
            xs += [W - w] if xs[-1] != W - w else []
            ys += [W - h] if ys[-1] != W - h else []
            out += [(x, y, x + w, y + h) for y in ys for x in xs]
    return np.array(out, dtype=np.float32)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def topk_diverse(boxes, scores, k, iou_thr=0.3):
    picked = []
    for i in np.argsort(-scores):
        if all(box_iou(boxes[i], boxes[j]) < iou_thr for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return picked


def valid_box(b, max_ar):
    x0, y0, x1, y1 = b
    if x0 < 0 or y0 < 0 or x1 > W or y1 > W or x1 - x0 < 4 or y1 - y0 < 4:
        return False
    return max(x1 - x0, y1 - y0) / min(x1 - x0, y1 - y0) <= max_ar * 1.02


def refine(wm, method, x_dev, s, box, max_ar, fp16, steps=(16, 8, 4, 2, 1), max_iter=40, mode="margin"):
    box = tuple(int(round(v)) for v in box)
    sc, ac = score_exact(wm, method, x_dev, s, [box], fp16, mode)
    best_s, best_a = float(sc[0]), float(ac[0])
    for st in steps:
        for _ in range(max_iter):
            x0, y0, x1, y1 = box
            cands = [(x0 - st, y0, x1, y1), (x0 + st, y0, x1, y1), (x0, y0 - st, x1, y1), (x0, y0 + st, x1, y1),
                     (x0, y0, x1 - st, y1), (x0, y0, x1 + st, y1), (x0, y0, x1, y1 - st), (x0, y0, x1, y1 + st),
                     (x0 - st, y0, x1 - st, y1), (x0 + st, y0, x1 + st, y1),
                     (x0, y0 - st, x1, y1 - st), (x0, y0 + st, x1, y1 + st)]
            cands = [c for c in cands if valid_box(c, max_ar)]
            if not cands:
                break
            sc, ac = score_exact(wm, method, x_dev, s, cands, fp16, mode)
            j = int(np.argmax(sc))
            if sc[j] <= best_s + 1e-6:
                break
            box, best_s, best_a = cands[j], float(sc[j]), float(ac[j])
    return box, best_s, best_a


def shrink_to_peak(wm, method, x_dev, s, box, best_s, best_a, max_ar, fp16, tol, steps=(8, 4, 2, 1), mode="margin"):
    """E2 (TrustMark/PixelSeal): growing a box past the true frame barely lowers the score, shrinking does.
    So after the ascent, pull each edge inwards while the score stays within a *relative* `tol` of the best.
    Use with the non-saturating "margin" score: with tanh, a saturated score lets the box shrink into the frame."""
    if tol <= 0:
        return box, best_s, best_a
    top = best_s
    for st in steps:
        changed = True
        while changed:
            changed = False
            x0, y0, x1, y1 = box
            cands = [c for c in ((x0 + st, y0, x1, y1), (x0, y0 + st, x1, y1), (x0, y0, x1 - st, y1),
                                 (x0, y0, x1, y1 - st)) if valid_box(c, max_ar)]
            if not cands:
                break
            sc, ac = score_exact(wm, method, x_dev, s, cands, fp16, mode)
            j = int(np.argmax(sc))
            if sc[j] >= top - tol * abs(top):
                box, best_s, best_a, changed = cands[j], float(sc[j]), float(ac[j]), True
                top = max(top, best_s)
    return box, best_s, best_a


def window_contrast(img, boxes, ring=8):
    """Log ratio of fine-noise energy inside each window vs a ring around it (integral images, vectorized).
    Small StegaStamp patches carry a squeezed, noisy residual: this ranked them with AUC ~0.70 on train."""
    g = img.astype(np.float32).mean(-1)
    E = (g - ndimage.median_filter(g, 3)) ** 2
    ii = np.pad(E, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x0, y0, x1, y1 = [boxes[:, i].astype(int) for i in range(4)]

    def S(a0, b0, a1, b1):
        return ii[b1, a1] - ii[b0, a1] - ii[b1, a0] + ii[b0, a0]
    ox0, oy0 = np.maximum(x0 - ring, 0), np.maximum(y0 - ring, 0)
    ox1, oy1 = np.minimum(x1 + ring, W), np.minimum(y1 + ring, W)
    s_in, a_in = S(x0, y0, x1, y1), (x1 - x0) * (y1 - y0)
    s_out, a_out = S(ox0, oy0, ox1, oy1), (ox1 - ox0) * (oy1 - oy0)
    ring_mean = (s_out - s_in) / np.maximum(a_out - a_in, 1)
    return np.log((s_in / a_in + 1e-3) / (ring_mean + 1e-3))


def frame_search(wm, method, img, msg, windows, a):
    x_dev = img_tensor(img).to(device())
    s = sign_vec(msg)
    keep = a.prefilter if a.prefilter >= 0 else DEFAULT_PREFILTER[method]
    if 0 < keep < len(windows):
        windows = windows[np.argsort(-window_contrast(img, windows))[:keep]]
    coarse = score_windows(wm, method, x_dev, s, windows, a.bs, a.fp16)
    cands = []
    for i in topk_diverse(windows, coarse, a.topk):
        box, sc, ac = refine(wm, method, x_dev, s, windows[i], MAX_AR[method], a.fp16, mode=a.refine_score)
        cands.append(dict(box=list(box), refine_score=sc, acc=ac, coarse=float(coarse[i])))
    best = max(cands, key=lambda c: c["refine_score"])
    tol = a.shrink_tol if a.shrink_tol >= 0 else DEFAULT_SHRINK_TOL[method]
    box, sc, ac = shrink_to_peak(wm, method, x_dev, s, tuple(best["box"]), best["refine_score"], best["acc"],
                                 MAX_AR[method], a.fp16, tol, mode=a.refine_score)
    soft, acc = score_exact(wm, method, x_dev, s, [box], a.fp16, "tanh")   # tanh units, comparable with the mode score
    best.update(box=list(box), refine_score=sc, soft=float(soft[0]), acc=float(acc[0]))
    best["coarse_max"] = float(coarse.max())
    best["area"] = (box[2] - box[0]) * (box[3] - box[1]) / W ** 2
    return best


def box_mask128(box):
    m = np.zeros((W, W), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    return downsample_mask(m)


# ------------------------------------------------------------------------------------- fw=True: hole maps
def presence_map(wm, method, img, msg):
    if method in ("pixelseal", "trustmark"):
        return decoder_maps(wm, method, img, sign_vec(msg))["dec_agree"]
    Y = img.astype(np.float32)
    return local_ncc(highpass(Y), highpass(embed_f(wm, img, msg) - Y), 16)


def hole_mask128(pmap, q, min_px=256):
    """1 = watermarked. The hole is the lowest-q quantile of the smoothed presence map (small blobs dropped)."""
    if q <= 0:
        return np.ones((128, 128), np.uint8)
    sm = ndimage.gaussian_filter(pmap, 4)
    hole = sm < np.quantile(sm, q)
    lab, n = ndimage.label(hole)
    if n:
        sizes = ndimage.sum(hole, lab, range(1, n + 1))
        hole = np.isin(lab, np.nonzero(sizes >= min_px)[0] + 1)
    return downsample_mask(~hole)


# ------------------------------------------------------------------------------------------------- driver
@torch.no_grad()
def global_soft(wm, method, items, fp16):
    out = []
    for i in range(0, len(items), 16):
        x = torch.cat([prep_to_res(img_tensor(it["image"]).to(device()), method) for it in items[i:i + 16]])
        lg = decode_at_res(wm, method, x, fp16)
        for it, l in zip(items[i:i + 16], lg):
            out.append(float(soft_acc(l[None], sign_vec(it["message"]))[0][0]))
    return np.array(out)


def decide_fw(g, thr, fr, margin, max_area=0.5):
    """Predicted full_watermark. Base rule: full-image score >= thr. Rescue (E1 errors were large fw=False frames):
    if a frame of at most `max_area` decodes at least `margin` better than the full image, call it fw=False."""
    if g < thr:
        return False
    if fr is not None and margin is not None and fr["area"] <= max_area and fr["soft"] >= g + margin:
        return False
    return True


def fw_true_mask(wm, method, it, q):
    if q <= 0:
        return np.ones((128, 128), np.uint8)
    return hole_mask128(presence_map(wm, method, it["image"], it["message"]), q)


def run_method(method, a):
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wm = load_wm(method, a.models_dir)
    windows = gen_windows(MAX_AR[method], a.sizes or DEFAULT_SIZES[method], a.stride_frac,
                          a.max_stride if a.max_stride > 0 else DEFAULT_MAX_STRIDE[method])
    keep = a.prefilter if a.prefilter >= 0 else DEFAULT_PREFILTER[method]
    note = f", {keep} decoded after the noise pre-filter" if 0 < keep < len(windows) else ""
    print(f"[{method}] {len(windows)} coarse windows per image{note}", flush=True)
    cache_path = out / f"fs_cache_{method}{f'_s{a.shard_idx}' if a.shard_n > 1 else ''}.json"   # per shard: no write races
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def search(it):
        if it["image_name"] not in cache:
            cache[it["image_name"]] = frame_search(wm, method, it["image"], it["message"], windows, a)
            cache_path.write_text(json.dumps(cache))
        return cache[it["image_name"]]

    # ---- train: calibrate mode threshold, mode rescue margin and hole quantile
    tr = LocalisationDataset(a.data, "train", methods=[method])
    tr_items = [tr[i] for i in range(len(tr))]
    fw = np.array([bool(it["full_watermark"]) for it in tr_items])
    g_tr = global_soft(wm, method, tr_items, a.fp16)
    mode_acc, thr = best_threshold_acc(g_tr, fw)
    print(f"[{method}] mode threshold {thr:.4f} -> train mode accuracy {mode_acc:.3f}", flush=True)

    margin, margin_scores = None, {}
    if a.search_below > thr:                       # only images with thr <= g < search_below can be rescued
        band = [i for i in range(len(tr_items)) if thr <= g_tr[i] < a.search_below]
        print(f"[{method}] rescue check: searching {len(band)} train images with {thr:.3f} <= g < {a.search_below}",
              flush=True)
        fr_tr = {i: search(tr_items[i]) for i in band}
        for mg in [None, 0.3, 0.2, 0.1, 0.05, 0.0]:
            pred = [decide_fw(g_tr[i], thr, fr_tr.get(i), mg) for i in range(len(tr_items))]
            margin_scores[str(mg)] = float(np.mean(np.array(pred) == fw))
        best_acc = max(margin_scores.values())
        # the first (most conservative) margin that reaches the best accuracy
        margin = next(None if k == "None" else float(k) for k, v in margin_scores.items() if v == best_acc)
        print(f"[{method}] rescue margin accuracies {margin_scores} -> margin={margin}", flush=True)

    q_best, q_scores = 0.0, {}
    if a.hole_q:
        maps = {it["image_name"]: presence_map(wm, method, it["image"], it["message"])
                for it in tr_items if it["full_watermark"]}
        for q in [0.0] + a.hole_q:
            q_scores[q] = float(np.mean([miou_image(hole_mask128(maps[it["image_name"]], q), it["mask128"])
                                         for it in tr_items if it["full_watermark"]]))
        q_best = max(q_scores, key=q_scores.get)
        print(f"[{method}] hole quantile on train fw=True: {q_scores} -> q={q_best}", flush=True)

    calib = dict(method=method, mode_thr=thr, mode_acc_train=mode_acc, rescue_margin=margin,
                 rescue_scores=margin_scores, hole_q=q_best, hole_q_scores=q_scores)

    def predict(it, g):
        fr = search(it) if g < a.search_below else None
        if decide_fw(g, thr, fr, margin):
            return fw_true_mask(wm, method, it, q_best), dict(pred_fw=True, g=float(g), **(fr or {}))
        return box_mask128(fr["box"]), dict(pred_fw=False, g=float(g), **fr)

    # ---- train evaluation (shard 0 only)
    if a.shard_idx == 0 and a.train_n != 0:
        sel = np.random.default_rng(0).permutation(len(tr_items))[:a.train_n if a.train_n > 0 else None]
        res = []
        for k, i in enumerate(sel):
            it = tr_items[i]
            p, info = predict(it, g_tr[i])
            m = it["mask"].astype(bool); x0, y0, x1, y1 = bbox(m)
            res.append(dict(name=it["image_name"], fw=bool(it["full_watermark"]),
                            is_rect=bool((not it["full_watermark"]) and m[y0:y1, x0:x1].all()),
                            gt_area=float(m.mean()), miou=miou_image(p, it["mask128"]), **info))
            if k % 20 == 0:
                print(f"[{method}] train {k}/{len(sel)}  running mIoU={np.mean([r['miou'] for r in res]):.4f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        print(f"[{method}] ===== TRAIN mIoU overall {np.mean([r['miou'] for r in res]):.4f} (n={len(res)}); "
              f"mode accuracy {np.mean([r['pred_fw'] == r['fw'] for r in res]):.3f}")
        groups = (("fw=True", lambda r: r["fw"]), ("fw=False rect", lambda r: r["is_rect"]),
                  ("fw=False object", lambda r: not r["fw"] and not r["is_rect"]))
        for name, f in groups:
            v = [r["miou"] for r in res if f(r)]
            print(f"   {name:16s} mIoU {np.mean(v) if v else float('nan'):.4f} (n={len(v)})")
        rects = [r for r in res if r["is_rect"] and not r["pred_fw"]]
        for lo, hi in ((0, .01), (.01, .03), (.03, .1), (.1, 1)):
            v = [r["miou"] for r in rects if lo <= r["gt_area"] < hi]
            if v:
                print(f"   rect area [{lo},{hi}): mIoU {np.mean(v):.3f} (n={len(v)})")
        found = [r for r in res if not r["pred_fw"] and "acc" in r]
        if found:
            accs = np.array([r["acc"] for r in found]); mi = np.array([r["miou"] for r in found])
            for t in (0.6, 0.7, 0.8, 0.9, 0.95):
                s_ = accs >= t
                print(f"   frame acc>={t}: n={s_.sum()} mIoU={mi[s_].mean() if s_.any() else float('nan'):.3f}"
                      f" | below: n={(~s_).sum()} mIoU={mi[~s_].mean() if (~s_).any() else float('nan'):.3f}")
        calib["train_eval"] = res
    (out / f"calib_{method}.json").write_text(json.dumps(calib, indent=1))

    # ---- test predictions
    if not (Path(a.data) / "test" / "metadata.json").exists():
        print("no test split found; skipping test predictions"); return
    te = LocalisationDataset(a.data, "test", methods=[method])
    te_items = [te[i] for i in range(len(te)) if i % a.shard_n == a.shard_idx]
    g_te = global_soft(wm, method, te_items, a.fp16)
    names, masks, info_all = [], [], []
    for k, (it, g) in enumerate(zip(te_items, g_te)):
        p, info = predict(it, g)
        names.append(it["image_name"]); masks.append(p.astype(np.uint8)); info_all.append(info)
        if k % 20 == 0:
            print(f"[{method}] test {k}/{len(te_items)}  {time.time() - t0:.0f}s", flush=True)
    tag = f"_s{a.shard_idx}" if a.shard_n > 1 else ""
    np.savez_compressed(out / f"pred_test_{method}{tag}.npz", names=np.array(names), masks=np.stack(masks))
    (out / f"pred_test_{method}{tag}.json").write_text(json.dumps(dict(zip(names, info_all))))
    n_fw = sum(i["pred_fw"] for i in info_all)
    print(f"[{method}] wrote {len(names)} test masks ({n_fw} predicted fw=True) in {time.time() - t0:.0f}s")


def merge(a):
    names, masks = [], []
    for f in sorted(Path(a.out).glob("pred_test_*.npz")):
        z = np.load(f); names += z["names"].tolist(); masks.append(z["masks"])
        print("merging", f.name, len(z["names"]))
    masks = np.concatenate(masks)
    assert len(names) == len(set(names)), "duplicate image names across prediction files"
    te = LocalisationDataset(a.data, "test") if (Path(a.data) / "test" / "metadata.json").exists() else None
    if te is not None:
        missing = set(m["image_name"] for m in te.meta) - set(names)
        assert not missing, f"{len(missing)} test images have no prediction (e.g. {list(missing)[:3]})"
    assert masks.shape[1:] == (128, 128) and set(np.unique(masks)) <= {0, 1}
    torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, a.submission)
    print(f"wrote {a.submission}: {len(names)} masks, {Path(a.submission).stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release"); ap.add_argument("--out", default="results")
    ap.add_argument("--models_dir", default=None); ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="window sizes (default: per method)")
    ap.add_argument("--stride_frac", type=float, default=0.5, help="window stride as a fraction of its size")
    ap.add_argument("--search_below", type=float, default=0.95,
                    help="frame-search every image whose full-image soft score is below this (mode rescue)")
    ap.add_argument("--shrink_tol", type=float, default=-1, help="relative; -1 = per-method default, 0 = off")
    ap.add_argument("--refine_score", choices=["margin", "tanh"], default="margin",
                    help="score maximized during edge refinement (margin does not saturate)")
    ap.add_argument("--max_stride", type=int, default=0, help="cap on the stride in px (0 = per-method default)")
    ap.add_argument("--prefilter", type=int, default=-1,
                    help="decode only the N windows with the highest noise contrast (-1 = per-method default, 0 = off)")
    ap.add_argument("--topk", type=int, default=5); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--hole_q", type=float, nargs="*", default=[0.02, 0.04, 0.06, 0.08, 0.1, 0.15],
                    help="hole-size quantiles tried on train (empty list disables hole maps)")
    ap.add_argument("--train_n", type=int, default=-1, help="train images to evaluate (-1 all, 0 none)")
    ap.add_argument("--shard", default="0/1", help="k/n: this job handles test images i with i %% n == k")
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--submission", default="submission.pt")
    a = ap.parse_args()
    a.shard_idx, a.shard_n = map(int, a.shard.split("/"))
    if a.merge:
        return merge(a)
    for method in a.methods:
        run_method(method, a)


if __name__ == "__main__":
    main()
