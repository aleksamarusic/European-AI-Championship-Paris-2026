"""End-to-end fine-tuned PixelSeal ConvNeXtV2-Tiny FPN with message conditioning, JND map, and Symmetric Soft-mIoU loss.

Why: Randomly-initialized pixel U-Nets stay at chance (~0.29 mIoU) on PixelSeal's 48 dB adversarially-trained residual.
PixelSeal's own pretrained detector backbone (wm.det.convnext, ConvNeXtV2-Tiny) ALREADY extracts features sensitive to
the watermark across 4 spatial scales (/4=128, /8=64, /16=32, /32=16 for 512px input). We:
  1. Keep a frozen copy of wm.det to compute exact message-agreement maps at native 256 and 512 scales, plus mode_256
     (0.9976 mode AUC) and PixelSeal's exact JND(x) heatmap.
  2. Fine-tune a trainable copy of wm.det.convnext (backbone lr=1e-4) with a multi-scale FPN decoder (head lr=1e-3)
     injecting the message maps + JND + mode_256 to output logits directly on the official 128x128 grid.
  3. Train on clean COCO covers embedded on-the-fly in GPU batches (exact uint8 rounding) + 118 real train images,
     optimizing 0.5*BCE + 0.5*SymmetricSoftIoU. Evaluates on the exact same 52 held-out val images as all other runs.
"""
import argparse
import copy
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
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


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


def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.GroupNorm(min(16, co), co),
        nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1, bias=False),
        nn.GroupNorm(min(16, co), co),
        nn.GELU(),
    )


class FrozenMsgExtractor:
    """Computes 12ch message-agreement + JND + mode_256 features at 16x16 and 2ch at 128x128."""

    def __init__(self, wm):
        self.wm = wm
        self.det = wm.det
        self.att = wm.model.attenuation
        self.lin_w = self.det.pixel_decoder.linear.weight.detach().float()
        self.lin_b = self.det.pixel_decoder.linear.bias.detach().float()

    @torch.no_grad()
    def __call__(self, x, s):
        # x: [B, 3, 512, 512] in [0, 1], s: [B, 256] in {-1, +1}
        B = x.shape[0]
        jnd = self.att.heatmaps(x)[:, 0:1].float() * 20.0  # [B, 1, 512, 512]
        jnd16 = F.avg_pool2d(jnd, 32)
        jnd128 = F.avg_pool2d(jnd, 4)

        # Native 256 pass -> 8x8
        x256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        emb256 = self.det.pixel_decoder.output_upscaling(self.det.convnext(self.det.preprocess(x256))).float()
        pp256 = torch.einsum("bchw,oc->bohw", emb256, self.lin_w) + self.lin_b[None, :, None, None]
        a256 = pp256[:, 1:] * s[:, :, None, None]  # [B, 256, 8, 8]
        mode256 = (torch.tanh(pp256[:, 1:].mean((2, 3))) * s).mean(1)  # [B]
        f256 = torch.stack([
            torch.tanh(a256).mean(1),
            (a256 > 0).float().mean(1),
            a256.mean(1) * 0.2,
            a256.std(1) * 0.2,
            pp256[:, 0] * 0.2,
        ], dim=1)  # [B, 5, 8, 8]
        f256_16 = F.interpolate(f256, size=(16, 16), mode="bilinear", align_corners=False)

        # Native-pixel 512 pass -> 16x16
        emb512 = self.det.pixel_decoder.output_upscaling(self.det.convnext(self.det.preprocess(x))).float()
        pp512 = torch.einsum("bchw,oc->bohw", emb512, self.lin_w) + self.lin_b[None, :, None, None]
        a512 = pp512[:, 1:] * s[:, :, None, None]  # [B, 256, 16, 16]
        f512 = torch.stack([
            torch.tanh(a512).mean(1),
            (a512 > 0).float().mean(1),
            a512.mean(1) * 0.2,
            a512.std(1) * 0.2,
            pp512[:, 0] * 0.2,
        ], dim=1)  # [B, 5, 16, 16]

        m16 = mode256.view(B, 1, 1, 1).expand(B, 1, 16, 16)
        m128 = mode256.view(B, 1, 1, 1).expand(B, 1, 128, 128)
        msg16 = torch.cat([f512, f256_16, m16, jnd16], dim=1)  # [B, 12, 16, 16]
        aux128 = torch.cat([jnd128, m128], dim=1)               # [B, 2, 128, 128]
        return msg16, aux128


class PixelSealFPN(nn.Module):
    def __init__(self, convnext_backbone):
        super().__init__()
        self.backbone = convnext_backbone  # ConvNeXtV2-Tiny: dims [96, 192, 384, 768]
        self.d3 = conv_block(768 + 12, 256)
        self.d2 = conv_block(256 + 384 + 12, 128)
        self.d1 = conv_block(128 + 192, 64)
        self.d0 = conv_block(64 + 96 + 12 + 2, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x, msg16, aux128):
        z = x * 2.0 - 1.0
        feats = []
        for i in range(4):
            z = self.backbone.downsample_layers[i](z)
            z = self.backbone.stages[i](z)
            feats.append(z)
        s0, s1, s2, s3 = feats  # 128, 64, 32, 16
        up = lambda t, sz: F.interpolate(t, size=(sz, sz), mode="bilinear", align_corners=False)  # noqa: E731
        p3 = self.d3(torch.cat([s3, msg16], 1))
        p2 = self.d2(torch.cat([up(p3, 32), s2, up(msg16, 32)], 1))
        p1 = self.d1(torch.cat([up(p2, 64), s1], 1))
        p0 = self.d0(torch.cat([up(p1, 128), s0, up(msg16, 128), aux128], 1))
        return self.head(p0)


@torch.no_grad()
def embed_batch_gpu(wm, x_batch, bits_batch):
    """Batched GPU PixelSeal embedding with exact uint8 quantization."""
    out = wm.model.embed(x_batch, msgs=bits_batch, is_video=False)["imgs_w"].clamp(0.0, 1.0)
    return torch.round(out * 255.0) / 255.0


@torch.no_grad()
def predict_items(model, fext, items, bs=8, flip_tta=False):
    model.eval()
    probs = []
    for i in range(0, len(items), bs):
        sub = items[i:i + bs]
        x = torch.from_numpy(np.stack([it["image"] for it in sub])).cuda().permute(0, 3, 1, 2).float() / 255.0
        s = torch.from_numpy(np.stack([2.0 * it["message"].astype(np.float32) - 1.0 for it in sub])).cuda()
        msg16, aux128 = fext(x, s)
        p = torch.sigmoid(model(x, msg16, aux128))
        if flip_tta:
            xf = x.flip(3)
            m16f, a128f = fext(xf, s)
            pf = torch.sigmoid(model(xf, m16f, a128f)).flip(3)
            p = 0.5 * (p + pf)
        probs.append(p[:, 0].cpu().numpy())
    model.train()
    return np.concatenate(probs)


def eval_mode_aware(probs, gts, fw_preds):
    best_s, best_t = -1.0, (0.5, 0.2)
    t_grid = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    f_grid = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    for tT in t_grid:
        sT = [miou_image((p >= tT).astype(np.uint8), g) for p, g, f in zip(probs, gts, fw_preds) if f]
        for tF in f_grid:
            sF = [miou_image((p >= tF).astype(np.uint8), g) for p, g, f in zip(probs, gts, fw_preds) if not f]
            sc = (sum(sT) + sum(sF)) / len(gts)
            if sc > best_s:
                best_s, best_t = sc, (tT, tF)
    sT = np.mean([miou_image((p >= best_t[0]).astype(np.uint8), g) for p, g, f in zip(probs, gts, fw_preds) if f])
    sF = np.mean([miou_image((p >= best_t[1]).astype(np.uint8), g) for p, g, f in zip(probs, gts, fw_preds) if not f])
    # Also single thr=0.5 for seg_combine compatibility
    s_single = {}
    for thr in (0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
        s_single[thr] = float(np.mean([miou_image((p >= thr).astype(np.uint8), g) for p, g in zip(probs, gts)]))
    best_single = max(s_single, key=s_single.get)
    return best_s, best_t, float(sT), float(sF), best_single, s_single[best_single]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=26.0)
    ap.add_argument("--eval_every", type=float, default=2.5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_coco", type=int, default=2500)
    ap.add_argument("--tag", default="ps_convnext_fpn")
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    t_start = time.time()
    out = ROOT / "results_seg" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    # Load real train & val (EXACT same val split as seg_train.py!)
    tr_all = LocalisationDataset(ROOT / "release", "train")
    all_items = [tr_all[i] for i in range(len(tr_all))]
    val_names = set(json.loads((ROOT / "results_seg" / "pixelseal" / "metrics.json").read_text())["val_names"])
    ps_items = [it for it in all_items if it["method"] == "pixelseal"]
    train_real = [it for it in ps_items if it["image_name"] not in val_names]
    val_real = sorted([it for it in ps_items if it["image_name"] in val_names], key=lambda it: it["image_name"])
    mode_json = json.loads((ROOT / "results_joba" / "mode_pixelseal_fp32.json").read_text())
    val_fw = [bool(mode_json["train"][it["image_name"]] >= mode_json["thr"]) for it in val_real]
    val_gts = [it["mask128"] for it in val_real]

    # Mask bank (excluding pixelseal val) + rect sizes
    bank, rect_sizes = [], []
    for it in all_items:
        if it["method"] == "pixelseal" and it["image_name"] in val_names:
            continue
        m = it["mask"]
        if is_rect(m):
            ys, xs = np.nonzero(m)
            w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            if w >= 16 and h >= 16:
                rect_sizes.append((w, h))
        else:
            bank.append(m)
    bank = np.stack(bank)

    # Load clean COCO covers into RAM
    coco_files = sorted((ROOT / "coco_dl" / "val2017").glob("*.jpg"))
    rng.shuffle(coco_files)
    covers = []
    for f in coco_files[:a.n_coco]:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        m = min(w, h)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((512, 512), Image.BICUBIC)
        covers.append(np.asarray(im, dtype=np.uint8))
    covers = np.stack(covers)
    print(f"[{a.tag}] loaded {len(covers)} COCO covers, {len(train_real)} real train, {len(val_real)} val, "
          f"{len(bank)} masks, {len(rect_sizes)} rects in {time.time() - t_start:.1f}s", flush=True)

    wm = load_watermark("pixelseal", device="cuda")
    for p in wm.model.parameters():
        p.requires_grad_(False)
    fext = FrozenMsgExtractor(wm)

    backbone = copy.deepcopy(wm.det.convnext).cuda()
    for p in backbone.parameters():
        p.requires_grad_(True)
    model = PixelSealFPN(backbone).cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init, weights_only=True))
        print(f"[{a.tag}] initialised from {a.init}", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": 1e-3, "weight_decay": 1e-4},
    ])

    real_imgs = np.stack([it["image"] for it in train_real])
    real_msgs = np.stack([2.0 * it["message"].astype(np.float32) - 1.0 for it in train_real])
    real_masks = np.stack([it["mask"] for it in train_real])

    budget = a.minutes * 60.0
    t0 = time.time()
    last_eval = t0
    it_n = 0
    best = dict(miou=-1.0)
    loss_ema = None

    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0:
            break
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * frac)) if frac > 0.04 else max(0.05, frac / 0.04)
        opt.param_groups[0]["lr"] = 1e-4 * lr_mult
        opt.param_groups[1]["lr"] = 1e-3 * lr_mult

        # Build mixed batch: 4 whole synth + 2 rect synth + 2 real (for bs=8)
        n_real = max(1, a.bs // 4)
        n_rect = max(1, a.bs // 4)
        n_whole = a.bs - n_real - n_rect

        xs_list, ss_list, ms_list = [], [], []

        # 1) Whole-image composites (batched GPU embed!)
        c_idx = rng.integers(len(covers), size=n_whole)
        c_np = covers[c_idx].copy()
        for k in range(n_whole):
            c_np[k] = np.rot90(c_np[k], int(rng.integers(4)))
            if rng.random() < 0.5:
                c_np[k] = c_np[k][:, ::-1]
        c_t = torch.from_numpy(np.ascontiguousarray(c_np)).cuda().permute(0, 3, 1, 2).float() / 255.0
        bits_t = torch.randint(0, 2, (n_whole, 256), device="cuda", dtype=torch.float32)
        w_t = embed_batch_gpu(wm, c_t, bits_t)
        m_np = []
        for _ in range(n_whole):
            mk = bank[rng.integers(len(bank))]
            mk = np.rot90(mk, int(rng.integers(4)))
            if rng.random() < 0.5:
                mk = mk[:, ::-1]
            m_np.append(np.ascontiguousarray(mk))
        m_t = torch.from_numpy(np.stack(m_np)).cuda()[:, None].float()
        xs_list.append(m_t * w_t + (1.0 - m_t) * c_t)
        ss_list.append(2.0 * bits_t - 1.0)
        ms_list.append(m_t)

        # 2) Rectangle composites (outpainting)
        for _ in range(n_rect):
            c = covers[rng.integers(len(covers))]
            c = np.rot90(c, int(rng.integers(4)))
            if rng.random() < 0.5:
                c = c[:, ::-1]
            bw, bh = rect_sizes[rng.integers(len(rect_sizes))]
            x0, y0 = int(rng.integers(0, 513 - bw)), int(rng.integers(0, 513 - bh))
            c1 = torch.from_numpy(np.ascontiguousarray(c)).cuda().permute(2, 0, 1)[None].float() / 255.0
            b1 = torch.randint(0, 2, (1, 256), device="cuda", dtype=torch.float32)
            crop_w = embed_batch_gpu(wm, c1[:, :, y0:y0 + bh, x0:x0 + bw], b1)
            y1 = c1.clone()
            y1[:, :, y0:y0 + bh, x0:x0 + bw] = crop_w
            m1 = torch.zeros((1, 1, 512, 512), device="cuda")
            m1[:, :, y0:y0 + bh, x0:x0 + bw] = 1.0
            xs_list.append(y1)
            ss_list.append(2.0 * b1 - 1.0)
            ms_list.append(m1)

        # 3) Real train images (with horizontal flip augmentation)
        r_idx = rng.integers(len(real_imgs), size=n_real)
        rx = torch.from_numpy(real_imgs[r_idx]).cuda().permute(0, 3, 1, 2).float() / 255.0
        rs = torch.from_numpy(real_msgs[r_idx]).cuda()
        rm = torch.from_numpy(real_masks[r_idx]).cuda()[:, None].float()
        if rng.random() < 0.5:
            rx = rx.flip(3)
            rm = rm.flip(3)
        xs_list.append(rx)
        ss_list.append(rs)
        ms_list.append(rm)

        x = torch.cat(xs_list, 0)
        s = torch.cat(ss_list, 0)
        t = F.avg_pool2d(torch.cat(ms_list, 0), 4)  # [B, 1, 128, 128]

        msg16, aux128 = fext(x, s)
        logits = model(x, msg16, aux128)
        loss = 0.5 * F.binary_cross_entropy_with_logits(logits, t) + 0.5 * sym_soft_iou_loss(logits, t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        loss_ema = loss.item() if loss_ema is None else 0.96 * loss_ema + 0.04 * loss.item()
        it_n += 1

        if time.time() - last_eval > a.eval_every * 60.0 or (time.time() - t0) >= budget:
            last_eval = time.time()
            pv = predict_items(model, fext, val_real, bs=8, flip_tta=False)
            m_mode, (tT, tF), sT, sF, b_sing, m_sing = eval_mode_aware(pv, val_gts, val_fw)
            print(f"[{a.tag}] it {it_n} ({(time.time() - t0)/60:.1f}m) loss {loss_ema:.4f} | "
                  f"val single {m_sing:.4f}@{b_sing} | mode-thr {m_mode:.4f}@({tT},{tF}) [fwT {sT:.3f} fwF {sF:.3f}]",
                  flush=True)
            if m_mode > best["miou"]:
                best = dict(miou=m_mode, single_miou=m_sing, single_thr=b_sing, thr_pair=(tT, tF),
                            by_group={"fw=True": sT, "fw=False": sF}, it=it_n)
                torch.save(model.state_dict(), out / "model.pt")

    torch.save(model.state_dict(), out / "last.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))

    # Final validation + test predictions with flip-TTA (continuous probabilities!)
    pv = predict_items(model, fext, val_real, bs=8, flip_tta=True)
    m_mode, (tT, tF), sT, sF, b_sing, m_sing = eval_mode_aware(pv, val_gts, val_fw)
    print(f"[{a.tag}] FINAL (flip-TTA): single {m_sing:.4f}@{b_sing} | "
          f"mode-thr {m_mode:.4f}@({tT},{tF}) [fwT {sT:.3f} fwF {sF:.3f}]", flush=True)

    te_ds = LocalisationDataset(ROOT / "release", "test", methods=["pixelseal"])
    te_items = [te_ds[i] for i in range(len(te_ds))]
    pt = predict_items(model, fext, te_items, bs=8, flip_tta=True)

    np.savez_compressed(out / "pred_val.npz", names=np.array([it["image_name"] for it in val_real]),
                        probs=(pv * 255.0).round().astype(np.uint8))
    np.savez_compressed(out / "pred_test.npz", names=np.array([it["image_name"] for it in te_items]),
                        probs=(pt * 255.0).round().astype(np.uint8))
    (out / "metrics.json").write_text(json.dumps(dict(
        method="pixelseal", tag=a.tag, val_miou=float(m_sing), val_miou_mode=float(m_mode),
        thr=float(b_sing), thresholds=[tT, tF], by_group={"fw=True": sT, "fw=False": sF},
        perfect_mode_val=0.4499, iterations=it_n, best_it=best.get("it"),
        val_names=sorted(val_names), minutes_total=(time.time() - t_start) / 60.0), indent=1))
    print(f"[{a.tag}] DONE in {(time.time() - t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
