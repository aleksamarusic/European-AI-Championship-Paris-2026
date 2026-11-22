"""End-to-end fine-tuned TrustMark ResNet-50 (wm.net) FPN with message conditioning, BandPass stem, and SymSoftIoU.

Same breakthrough paradigm as ps_convnext_fpn.py:
  1. Frozen wm.net computes message-agreement maps at native 245px (8x8) and 512px (16x16), plus exact mode_245.
  2. Trainable copy of TrustMark's pretrained ResNet-50 (layer1..layer4 -> 128, 64, 32, 16) + BandPass(x) stem at 128x128
     + FPN decoder injecting message maps + mode_245 -> 128x128 mask logits.
  3. Trained on pure clean COCO covers embedded with TrustMark + 118 real TrustMark train images, using
     0.4*BCE + 0.6*SymmetricSoftIoULoss. Evaluates on the exact 52 held-out TrustMark val images + flip-TTA.
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
sys.path.insert(0, str(ROOT / "experiments"))
from seg_train import BandPass, evaluate, group_of, is_rect, load_items  # noqa: E402
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


def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.GroupNorm(min(16, co), co),
        nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1, bias=False),
        nn.GroupNorm(min(16, co), co),
        nn.GELU(),
    )


class FrozenTMMsgExtractor:
    """Computes 9ch message-agreement + mode_245 features at 16x16 from frozen TrustMark ResNet-50."""

    def __init__(self, wm):
        self.net = wm.net
        self.res = wm.resolution  # 245
        self.fc_w = self.net.fc.weight.detach().float()
        self.fc_b = self.net.fc.bias.detach().float()

    def _layer4(self, x):
        n = self.net
        z = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        z = n.layer1(z); z = n.layer2(z); z = n.layer3(z); z = n.layer4(z)
        return z

    @torch.no_grad()
    def __call__(self, x, s):
        # x: [B, 3, 512, 512] in [0,1], s: [B, 100] in {-1,+1}
        B = x.shape[0]
        # Native 245 pass -> 8x8
        x245 = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True) * 2.0 - 1.0
        l4_245 = self._layer4(x245).float()
        pp245 = torch.einsum("bchw,oc->bohw", l4_245, self.fc_w) + self.fc_b[None, :, None, None]
        a245 = pp245 * s[:, :, None, None]
        mode245 = (torch.tanh(pp245.mean((2, 3))) * s).mean(1)  # [B]
        f245 = torch.stack([
            torch.tanh(a245).mean(1),
            (a245 > 0).float().mean(1),
            a245.mean(1) * 0.2,
            a245.std(1) * 0.2,
        ], 1)
        f245_16 = F.interpolate(f245, size=(16, 16), mode="bilinear", align_corners=False)

        # 512 pass -> 16x16
        l4_512 = self._layer4(x * 2.0 - 1.0).float()
        pp512 = torch.einsum("bchw,oc->bohw", l4_512, self.fc_w) + self.fc_b[None, :, None, None]
        a512 = pp512 * s[:, :, None, None]
        f512 = torch.stack([
            torch.tanh(a512).mean(1),
            (a512 > 0).float().mean(1),
            a512.mean(1) * 0.2,
            a512.std(1) * 0.2,
        ], 1)

        m16 = mode245.view(B, 1, 1, 1).expand(B, 1, 16, 16)
        return torch.cat([f512, f245_16, m16], 1)  # [B, 9, 16, 16]


class TrustMarkResNetFPN(nn.Module):
    def __init__(self, resnet_backbone):
        super().__init__()
        self.backbone = resnet_backbone  # ResNet-50: l1=256(128), l2=512(64), l3=1024(32), l4=2048(16)
        self.band = BandPass(gain=25.0)
        self.bp_stem = nn.Sequential(
            nn.Conv2d(12, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
        )  # 512 -> 128
        self.d4 = conv_block(2048 + 9, 256)
        self.d3 = conv_block(256 + 1024 + 9, 128)
        self.d2 = conv_block(128 + 512, 64)
        self.d1 = conv_block(64 + 256 + 64 + 9, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x, msg16):
        n = self.backbone
        z = n.maxpool(n.relu(n.bn1(n.conv1(x * 2.0 - 1.0))))
        l1 = n.layer1(z)   # 128x128
        l2 = n.layer2(l1)  # 64x64
        l3 = n.layer3(l2)  # 32x32
        l4 = n.layer4(l3)  # 16x16
        bp128 = self.bp_stem(self.band(x))
        up = lambda t, sz: F.interpolate(t, size=(sz, sz), mode="bilinear", align_corners=False)  # noqa: E731
        p4 = self.d4(torch.cat([l4, msg16], 1))
        p3 = self.d3(torch.cat([up(p4, 32), l3, up(msg16, 32)], 1))
        p2 = self.d2(torch.cat([up(p3, 64), l2], 1))
        p1 = self.d1(torch.cat([up(p2, 128), l1, bp128, up(msg16, 128)], 1))
        return self.head(p1)


@torch.no_grad()
def predict_items(model, fext, items, bs=8, flip_tta=True):
    model.eval()
    out = []
    for i in range(0, len(items), bs):
        sub = items[i:i + bs]
        x = torch.from_numpy(np.stack([it["image"] for it in sub])).cuda().permute(0, 3, 1, 2).float() / 255.0
        s = torch.from_numpy(np.stack([2.0 * it["message"].astype(np.float32) - 1.0 for it in sub])).cuda()
        m16 = fext(x, s)
        p = torch.sigmoid(model(x, m16))
        if flip_tta:
            xf = x.flip(3)
            m16f = fext(xf, s)
            pf = torch.sigmoid(model(xf, m16f)).flip(3)
            p = 0.5 * (p + pf)
        out.append(p[:, 0].cpu().numpy())
    model.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--eval_every", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_coco", type=int, default=800)
    ap.add_argument("--tag", default="tm_resnet50_fpn")
    ap.add_argument("--seed", type=int, default=7)
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
    val_names = set(json.loads((ROOT / "results_seg" / "trustmark" / "metrics.json").read_text())["val_names"])
    real = [it for it in tr_all if it["method"] == "trustmark"]
    train_real = [it for it in real if it["image_name"] not in val_names]
    val_real = sorted([it for it in real if it["image_name"] in val_names], key=lambda it: it["image_name"])
    val_gts = [it["mask128"] for it in val_real]
    val_groups = [group_of(it) for it in val_real]

    bank, rect_sizes = [], []
    for it in tr_all:
        if it["method"] == "trustmark" and it["image_name"] in val_names:
            continue
        m = it["mask"]
        if is_rect(m):
            if it["method"] == "trustmark":
                ys, xs = np.nonzero(m)
                rect_sizes.append((int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        else:
            bank.append(m)
    bank = np.stack(bank)

    files = sorted((ROOT / "coco_dl" / "val2017").glob("*.jpg"))
    rng.shuffle(files)
    covers = []
    for f in files[:a.n_coco]:
        im = Image.open(f).convert("RGB"); w, h = im.size; m = min(w, h)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((512, 512), Image.BICUBIC)
        covers.append(np.asarray(im, dtype=np.uint8))
    covers = np.stack(covers)

    wm = load_watermark("trustmark", device="cuda")
    for p in wm.net.parameters():
        p.requires_grad_(False)
    fext = FrozenTMMsgExtractor(wm)

    # Pre-embed clean COCO covers WITH their known messages!
    t_e = time.time()
    W, W_msgs = [], []
    for c in covers:
        msg = rng.integers(0, 2, wm.nbits)
        W.append(wm.embed(c, msg))
        W_msgs.append((2.0 * msg - 1.0).astype(np.float32))
    W = np.stack(W); W_msgs = np.stack(W_msgs)
    RY, RM, R_msgs = [], [], []
    for c in covers[:min(len(covers), 500)]:
        bw, bh = rect_sizes[rng.integers(len(rect_sizes))]
        x0, y0 = int(rng.integers(0, 513 - bw)), int(rng.integers(0, 513 - bh))
        msg = rng.integers(0, 2, wm.nbits)
        try:
            y = c.copy()
            y[y0:y0 + bh, x0:x0 + bw] = wm.embed(np.ascontiguousarray(c[y0:y0 + bh, x0:x0 + bw]), msg)
            m = np.zeros((512, 512), np.uint8); m[y0:y0 + bh, x0:x0 + bw] = 1
            RY.append(y); RM.append(m); R_msgs.append((2.0 * msg - 1.0).astype(np.float32))
        except Exception:
            continue
    RY, RM, R_msgs = np.stack(RY), np.stack(RM), np.stack(R_msgs)
    print(f"[{a.tag}] embedded {len(W)} covers + {len(RY)} rects with msgs in {time.time()-t_e:.1f}s", flush=True)

    backbone = copy.deepcopy(wm.net).cuda()
    for p in backbone.parameters():
        p.requires_grad_(True)
    model = TrustMarkResNetFPN(backbone).cuda()

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": 1e-3, "weight_decay": 1e-4},
    ])

    real_imgs = np.stack([it["image"] for it in train_real])
    real_msgs = np.stack([2.0 * it["message"].astype(np.float32) - 1.0 for it in train_real])
    real_masks = np.stack([it["mask"] for it in train_real])

    budget = a.minutes * 60.0
    t0 = time.time(); last_eval = t0; it_n = 0; best = dict(miou=-1.0); loss_ema = None

    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0:
            break
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * frac)) if frac > 0.05 else max(0.05, frac / 0.05)
        opt.param_groups[0]["lr"] = 1e-4 * lr_mult
        opt.param_groups[1]["lr"] = 1e-3 * lr_mult

        xs, ss, ms = [], [], []
        kinds = rng.choice(3, a.bs, p=[0.5, 0.2, 0.3])
        for k in kinds:
            if k == 0:
                idx = int(rng.integers(len(W)))
                mk = bank[int(rng.integers(len(bank)))]
                mk = np.rot90(mk, int(rng.integers(4)))
                if rng.random() < 0.5: mk = mk[:, ::-1]
                mk = np.ascontiguousarray(mk)
                y = np.where(mk[..., None].astype(bool), W[idx], covers[idx])
                s = W_msgs[idx]
            elif k == 1:
                idx = int(rng.integers(len(RY)))
                y, mk, s = RY[idx], RM[idx], R_msgs[idx]
            else:
                idx = int(rng.integers(len(real_imgs)))
                y, mk, s = real_imgs[idx], real_masks[idx], real_msgs[idx]
            if rng.random() < 0.5:
                y = np.ascontiguousarray(y[:, ::-1]); mk = np.ascontiguousarray(mk[:, ::-1])
            xs.append(y); ss.append(s); ms.append(mk)

        x = torch.from_numpy(np.stack(xs)).cuda().permute(0, 3, 1, 2).float() / 255.0
        s = torch.from_numpy(np.stack(ss)).cuda()
        t = F.avg_pool2d(torch.from_numpy(np.stack(ms)).cuda()[:, None].float(), 4)

        m16 = fext(x, s)
        logits = model(x, m16)
        loss = 0.4 * F.binary_cross_entropy_with_logits(logits, t) + 0.6 * sym_soft_iou_loss(logits, t)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        loss_ema = loss.item() if loss_ema is None else 0.96 * loss_ema + 0.04 * loss.item()
        it_n += 1

        # Refresh 2 covers every 15 iters with brand new messages & flips
        if it_n % 15 == 0:
            for _ in range(2):
                i = int(rng.integers(len(W)))
                c = np.rot90(covers[int(rng.integers(len(covers)))], int(rng.integers(4)))
                if rng.random() < 0.5: c = c[:, ::-1]
                c = np.ascontiguousarray(c)
                msg = rng.integers(0, 2, wm.nbits)
                covers[i] = c; W[i] = wm.embed(c, msg); W_msgs[i] = (2.0 * msg - 1.0).astype(np.float32)

        if time.time() - last_eval > a.eval_every * 60.0 or (time.time() - t0) >= budget:
            last_eval = time.time()
            pv = predict_items(model, fext, val_real, bs=8, flip_tta=False)
            thr, miou, by_group, _ = evaluate(pv, val_gts, val_groups)
            gtxt = " ".join(f"{k} {v:.3f}" for k, v in by_group.items())
            print(f"[{a.tag}] it {it_n} ({(time.time()-t0)/60:.1f}m) loss {loss_ema:.4f} | val {miou:.4f} @{thr} ({gtxt})", flush=True)
            if miou > best["miou"]:
                best = dict(miou=miou, thr=thr, by_group=by_group, it=it_n)
                torch.save(model.state_dict(), out / "model.pt")

    torch.save(model.state_dict(), out / "last.pt")
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    pv = predict_items(model, fext, val_real, bs=8, flip_tta=True)
    thr, miou, by_group, sweep = evaluate(pv, val_gts, val_groups)
    print(f"[{a.tag}] FINAL (flip-TTA) val {miou:.4f} @{thr} {by_group}", flush=True)

    te = load_items(ROOT / "release", "test", ["trustmark"])
    pt = predict_items(model, fext, te, bs=8, flip_tta=True)
    np.savez_compressed(out / "pred_val.npz", names=np.array([it["image_name"] for it in val_real]),
                        probs=(pv * 255.0).round().astype(np.uint8))
    np.savez_compressed(out / "pred_test.npz", names=np.array([it["image_name"] for it in te]),
                        probs=(pt * 255.0).round().astype(np.uint8))
    oracle = float(np.mean([miou_image(np.full((128, 128), int(it["full_watermark"]), np.uint8), it["mask128"])
                            for it in val_real]))
    (out / "metrics.json").write_text(json.dumps(dict(
        method="trustmark", tag=a.tag, val_miou=float(miou), thr=float(thr), by_group=by_group,
        thr_sweep=sweep, perfect_mode_val=oracle, iterations=it_n, best_it=best.get("it"),
        val_names=sorted(val_names), minutes_total=(time.time() - t_start) / 60.0), indent=1))
    print(f"[{a.tag}] DONE in {(time.time()-t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
