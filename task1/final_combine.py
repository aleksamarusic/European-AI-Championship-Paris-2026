"""Final principled ensemble + mode-aware post-processing + hybrid frame-search combiner.

Features:
  1. Per-method weighted probability ensembling (only over runs with continuous probabilities).
  2. Mode-specific probability thresholds (thr_T for pred_fw=True, thr_F for pred_fw=False) using exact fp32 mode scores.
  3. Non-empty recovery on pred_fw=False: if pred is empty and p.max() >= peak_min, take (p >= 0.5 * p.max()).
  4. Mode area clamp on pred_fw=False: never allow predicted area > 0.50 (fixes TrustMark's 3c541165b889-style errors).
  5. Morphological speck/hole removal (min_cells).
  6. Hybrid frame-search box override (TrustMark fs_trustmark, PixelSeal fs_pixelseal_fp32) when bit acc >= tau.
  7. Strict validation check vs current best; verifies 1020 masks, bool [1020, 128, 128], <= 20 MB, weights_only=True.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402


def is_rect(m):
    ys, xs = np.nonzero(m)
    return len(ys) > 0 and m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all()


def clean_mask(m, min_cells):
    if min_cells <= 0:
        return m
    m = m.astype(bool)
    for val in (True, False):
        lab, n = ndimage.label(m == val, structure=np.ones((3, 3)))
        if n:
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            small = np.isin(lab, np.nonzero(sizes < min_cells)[0] + 1)
            m = np.where(small, not val, m)
    return m.astype(np.uint8)


def apply_post(p, fw_pred, thr_T, thr_F, min_cells=0, peak_min=0.04, clamp_false=True, snap_rect=False):
    thr = thr_T if fw_pred else thr_F
    m = (p >= thr).astype(np.uint8)
    # Mode clamp for fw_pred=False: never predict > 50% watermarked
    if clamp_false and (not fw_pred) and m.mean() > 0.50:
        # Keep top 45% or raise threshold
        q = float(np.quantile(p, 0.55))
        m = (p >= max(thr, q)).astype(np.uint8)
    # Non-empty recovery for fw_pred=False when a clear localized peak exists
    if (not fw_pred) and m.sum() == 0 and peak_min > 0 and p.max() >= peak_min:
        m = (p >= 0.5 * float(p.max())).astype(np.uint8)
    if min_cells > 0:
        m_c = clean_mask(m, min_cells)
        # Don't wipe out a recovered small mask completely
        if m_c.sum() > 0 or fw_pred:
            m = m_c
    # Optional rectangle snapping for fw_pred=False when largest component is already very rectangular
    if snap_rect and (not fw_pred) and m.sum() >= 16:
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 1:
            ys, xs = np.nonzero(m)
            area_box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            if m.sum() / area_box >= 0.85:
                m2 = np.zeros_like(m)
                m2[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
                m = m2
    return m


def load_run_probs(tag):
    run = ROOT / "results_seg" / tag
    if not (run / "metrics.json").exists():
        return None
    met = json.loads((run / "metrics.json").read_text())
    pv = np.load(run / "pred_val.npz")
    pt = np.load(run / "pred_test.npz")
    val = {n: p.astype(np.float32) / 255.0 for n, p in zip(pv["names"].tolist(), pv["probs"])}
    test = {n: p.astype(np.float32) / 255.0 for n, p in zip(pt["names"].tolist(), pt["probs"])}
    return dict(tag=tag, met=met, val=val, test=test)


def load_mode(method):
    p = ROOT / "results_joba" / (f"mode_{method}_fp32.json" if method == "pixelseal" else f"mode_{method}.json")
    d = json.loads(p.read_text())
    return d


def load_fs_trustmark():
    calib = ROOT / "results_joba" / "fs_trustmark" / "calib_trustmark.json"
    summary = ROOT / "results_joba" / "job_a_summary.json"
    npzs = sorted((ROOT / "results_joba").glob("fs_trustmark*/pred_test_trustmark*.npz"))
    rows = {r["name"]: r for r in json.loads(calib.read_text()).get("train_eval", [])}
    tau = json.loads(summary.read_text())["per_method"]["trustmark"]["tau"]
    test = {}
    for f in npzs:
        z = np.load(f)
        info = json.loads(f.with_suffix(".json").read_text())
        test.update({n: (m.astype(np.uint8), info[n]) for n, m in zip(z["names"].tolist(), z["masks"])})
    return dict(rows=rows, tau=tau, test=test)


def load_fs_pixelseal():
    d = ROOT / "results_joba" / "fs_pixelseal_fp32"
    if not (d / "train_rows.json").exists() or not (d / "test_rows.json").exists():
        return None
    tr_rows = json.loads((d / "train_rows.json").read_text())
    te_rows = json.loads((d / "test_rows.json").read_text())
    test = {}
    if (d / "pred_test.npz").exists():
        z = np.load(d / "pred_test.npz")
        for n, m in zip(z["names"].tolist(), z["masks"]):
            test[n] = (m.astype(np.uint8), te_rows[n])
    return dict(rows=tr_rows, test=test)


def blend_runs(runs, weights):
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    vn = list(runs[0]["val"].keys())
    tn = list(runs[0]["test"].keys())
    val = {n: sum(wi * r["val"][n] for wi, r in zip(w, runs)) for n in vn}
    test = {n: sum(wi * r["test"][n] for wi, r in zip(w, runs)) for n in tn}
    return val, test


def eval_config(val_probs, names, gt, fw_map, thr_T, thr_F, min_cells, peak_min, clamp_false, snap_rect, fs=None, tau=None):
    scores = []
    for n in names:
        if fs is not None and tau is not None and n in fs["rows"]:
            r = fs["rows"][n]
            if (not r["pred_fw"]) and r.get("acc", 0.0) >= tau:
                scores.append(r["miou"])
                continue
        m = apply_post(val_probs[n], fw_map[n], thr_T, thr_F, min_cells, peak_min, clamp_false, snap_rect)
        scores.append(miou_image(m, gt[n]))
    return float(np.mean(scores)), scores


def optimize_method(method, candidate_tags, gt, meta, mode, fs=None):
    avail = [load_run_probs(t) for t in candidate_tags]
    avail = [r for r in avail if r is not None]
    assert avail, f"No available runs for {method}"
    names = avail[0]["met"]["val_names"]
    fw_map = {n: bool(mode["train"][n] >= mode["thr"]) for n in names}

    print(f"\n================ {method.upper()} (val n={len(names)}) ================")
    for r in avail:
        print(f"  available: {r['tag']:26s} recorded_val={r['met']['val_miou']:.4f}")

    # Build candidate probability blends (each single run + simple averages of top pairs/triples)
    blends = []
    for r in avail:
        blends.append((r["tag"], r["val"], r["test"]))
    # Pairwise & triple blends among continuous runs (exclude binarized ps_dechead_synth_big from soft avg unless alone)
    cont = [r for r in avail if not (r["tag"] in ("ps_dechead_synth_big", "ps_dechead_synth", "ps_dechead_cv", "ps_dechead_synth_huge"))]
    for i in range(len(cont)):
        for j in range(i + 1, len(cont)):
            for wi, wj in ((0.5, 0.5), (0.7, 0.3), (0.3, 0.7)):
                vp, tp = blend_runs([cont[i], cont[j]], [wi, wj])
                blends.append((f"{wi:.1f}*{cont[i]['tag']}+{wj:.1f}*{cont[j]['tag']}", vp, tp))
    if len(cont) >= 3:
        vp, tp = blend_runs(cont[:3], [1.0, 1.0, 1.0])
        blends.append(("equal3:" + "+".join(r["tag"] for r in cont[:3]), vp, tp))
        vp, tp = blend_runs(cont[:3], [0.5, 0.25, 0.25])
        blends.append(("w3_50_25_25:" + "+".join(r["tag"] for r in cont[:3]), vp, tp))
    if len(cont) >= 4:
        vp, tp = blend_runs(cont[:4], [1.0, 1.0, 1.0, 1.0])
        blends.append(("equal4:" + "+".join(r["tag"] for r in cont[:4]), vp, tp))
        vp, tp = blend_runs(cont[:4], [0.4, 0.3, 0.15, 0.15])
        blends.append(("w4_40_30_15_15:" + "+".join(r["tag"] for r in cont[:4]), vp, tp))
    # Also allow blending binarized ps_dechead_synth_big/huge with continuous PixelSeal runs if useful
    if method == "pixelseal" and len(avail) >= 2 and len(cont) >= 1:
        bin_runs = [r for r in avail if r not in cont]
        for br in bin_runs[:1]:
            for cr in cont:
                vp, tp = blend_runs([cr, br], [0.6, 0.4])
                blends.append((f"0.6*{cr['tag']}+0.4*{br['tag']}", vp, tp))

    # Determine tau grid for fs
    if fs is None:
        taus = [None]
    elif method == "trustmark":
        taus = [fs["tau"], 0.80, 0.85, 1.01]
    else:
        taus = [0.72, 0.75, 0.80, 0.85, 1.01]

    tT_grid = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    tF_grid = (0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)

    best_T = None
    best_F = None
    for bname, vp, tp in blends:
        # 1) Optimize fw_pred=True independently!
        for tT in tT_grid:
            for mc in (0, 16, 32):
                vals = [miou_image(apply_post(vp[n], True, tT, 0.5, mc, 0.0, False, False), gt[n])
                        for n in names if fw_map[n]]
                scT = float(np.mean(vals))
                if best_T is None or scT > best_T[0] + 1e-6:
                    best_T = (scT, bname, tT, mc, vp, tp)

        # 2) Optimize fw_pred=False independently!
        for tau in taus:
            for tF in tF_grid:
                for mc in (0, 16):
                    for pk in (0.0, 0.04):
                        vals = []
                        for n in names:
                            if fw_map[n]:
                                continue
                            if fs is not None and tau is not None and n in fs["rows"]:
                                r = fs["rows"][n]
                                if (not r["pred_fw"]) and r.get("acc", 0.0) >= tau:
                                    vals.append(r["miou"])
                                    continue
                            m = apply_post(vp[n], False, 0.5, tF, mc, pk, True, False)
                            vals.append(miou_image(m, gt[n]))
                        scF = float(np.mean(vals))
                        if best_F is None or scF > best_F[0] + 1e-6:
                            best_F = (scF, bname, tF, mc, pk, tau, vp, tp)

    scT, bnameT, tT, mcT, vpT, tpT = best_T
    scF, bnameF, tF, mcF, pkF, tauF, vpF, tpF = best_F

    sc_list = []
    for n in names:
        if fw_map[n]:
            sc_list.append(miou_image(apply_post(vpT[n], True, tT, 0.5, mcT, 0.0, False, False), gt[n]))
        else:
            if fs is not None and tauF is not None and n in fs["rows"]:
                r = fs["rows"][n]
                if (not r["pred_fw"]) and r.get("acc", 0.0) >= tauF:
                    sc_list.append(r["miou"])
                    continue
            sc_list.append(miou_image(apply_post(vpF[n], False, 0.5, tF, mcF, pkF, True, False), gt[n]))
    sc = float(np.mean(sc_list))

    # Group breakdown
    grp_scores = {}
    for n, s_val in zip(names, sc_list):
        g_lbl = "fw=True" if meta[n]["full_watermark"] else ("rect" if is_rect(gt[n]) else "object")
        grp_scores.setdefault(g_lbl, []).append(s_val)
    gtxt = " ".join(f"{k}={np.mean(v):.4f}" for k, v in sorted(grp_scores.items()))
    print(f"  BEST [{method}]: val={sc:.4f} ({gtxt})")
    print(f"    fw=True  ({scT:.4f}): blend={bnameT} | tT={tT} mc={mcT}")
    print(f"    fw=False ({scF:.4f}): blend={bnameF} | tF={tF} mc={mcF} pk={pkF} fs_tau={tauF}")

    # Generate test masks
    test_masks = {}
    n_box = 0
    for n in tpT:
        fw_t = bool(mode["test"][n] >= mode["thr"])
        if fw_t:
            test_masks[n] = apply_post(tpT[n], True, tT, 0.5, mcT, 0.0, False, False)
        else:
            if fs is not None and tauF is not None and n in fs["test"]:
                bm, info = fs["test"][n]
                if (not info["pred_fw"]) and info.get("acc", 0.0) >= tauF:
                    test_masks[n] = bm
                    n_box += 1
                    continue
            test_masks[n] = apply_post(tpF[n], False, 0.5, tF, mcF, pkF, True, False)
    print(f"    test: {len(test_masks)} masks ({n_box} fs boxes), mean wm area={np.mean([m.mean() for m in test_masks.values()]):.3f}")
    return sc, test_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission_seg6.pt")
    ap.add_argument("--base", default="submission_seg5.pt")
    a = ap.parse_args()

    meta = {m["image_name"]: m for m in json.loads((ROOT / "release" / "train" / "metadata.json").read_text())}
    z = np.load(ROOT / "release" / "train" / "masks_128.npz")
    gt = dict(zip(z["names"].tolist(), z["masks"]))

    candidates = {
        "stegastamp": ["ss_full512", "ss_fresh_band_tta", "stegastamp_tta", "ss_coco_tta"],
        "trustmark": ["tm_resnet50_fpn", "tm_full512", "tm_coco_ttaflip", "tm_fresh_band2_ttaflip"],
        "pixelseal": ["ps_convnext_fpn", "ps_convnext_fpn_d4", "ps_convnext_fpn_s1", "ps_convnext_fpn_s1_tta", "ps_dechead_v2", "ps_dechead_synth_big"],
    }
    fs_map = {
        "stegastamp": None,
        "trustmark": load_fs_trustmark(),
        "pixelseal": load_fs_pixelseal(),
    }

    all_masks = {}
    tot = []
    for method in ("pixelseal", "trustmark", "stegastamp"):
        sc, tm = optimize_method(method, candidates[method], gt, meta, load_mode(method), fs_map[method])
        all_masks.update(tm)
        tot.append(sc)
    print(f"\n=== OVERALL VALIDATION MEAN mIoU: {np.mean(tot):.4f} ===")

    base = torch.load(ROOT / a.base, weights_only=True)
    names = list(base["image_names"])
    test_names = {m["image_name"] for m in LocalisationDataset(ROOT / "release", "test").meta}
    assert set(names) == test_names and set(all_masks) == test_names
    t = torch.from_numpy(np.stack([all_masks[n] for n in names]).astype(bool))
    out_path = ROOT / a.out
    torch.save({"image_names": names, "masks": t}, out_path)
    sz = out_path.stat().st_size / 1e6
    assert sz <= 20.0, f"{sz:.2f} MB > 20 MB"
    chk = torch.load(out_path, weights_only=True)
    assert chk["masks"].shape == (len(names), 128, 128) and chk["masks"].dtype == torch.bool
    diff = (chk["masks"] != base["masks"]).float().mean().item()
    print(f"Wrote {out_path}: {len(names)} masks, {sz:.2f} MB, mean_area={t.float().mean():.3f}, cell_diff_vs_{a.base}={diff:.4f}")


if __name__ == "__main__":
    main()
