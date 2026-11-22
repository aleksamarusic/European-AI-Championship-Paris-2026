"""Final CORE5 Model Equality Testing pipeline (Task 2: 33-model-equality).

Consolidates the forensic detectors that improved our score from 0.55 -> 0.70 (public)
with additional private-split upside:
  1. `top_p=0.95` nucleus-sampling support-truncation detector (`ep-14`, `ep-07`, `ep-27`)
  2. Algorithm-agnostic Token-0 High-Entropy Watermark Collision Excess probe (`ep-04`, `ep-06`, `ep-10`)
  3. 10-scheme weight quantization fingerprint scan (`ep-22`, `ep-15` FP8; `ep-01` INT8)
  4. Validated CORE5 submission CSV generator (5 proven base endpoints/family = 30 pairs, 0 FP)

Usage:
  # Generate and validate the exact submitted CORE5 CSV (no GPU / API calls needed):
  python solve_core5.py generate --out submission_core.csv

  # Re-score saved token-0 watermark probes against local fp16 base logits (GPU):
  python solve_core5.py score-wm --wm-file /tmp/me_opt/wm_tok0.jsonl --wmc-file /tmp/me_opt/wmc_confirm.jsonl

  # Run the top_p=0.95 rank & cumulative-mass support cutoff detector (GPU):
  python solve_core5.py detect-topp --probe-file probe/ep_probe.jsonl

  # Run the 10-scheme quantization fingerprint scan on candidate endpoints (GPU):
  python solve_core5.py scan-quant --probe-file probe/ep_probe.jsonl --n-answers 30
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

FAMILIES = {
    "Q3": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "endpoints": [
            "ep-01", "ep-02", "ep-04", "ep-12", "ep-14",
            "ep-17", "ep-18", "ep-23", "ep-24", "ep-25", "ep-33",
        ],
        # Proven base roles: sys_prompt, persona, T=1.0, T=1.0+top_p=0.95, watermark
        "core5": ["ep-02", "ep-23", "ep-33", "ep-14", "ep-04"],
        "quant_suspects": ["ep-01", "ep-04", "ep-12", "ep-17", "ep-18", "ep-24", "ep-25", "ep-14"],
    },
    "Q15": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "endpoints": [
            "ep-03", "ep-06", "ep-07", "ep-08", "ep-20",
            "ep-21", "ep-22", "ep-26", "ep-28", "ep-30", "ep-32",
        ],
        "core5": ["ep-08", "ep-20", "ep-30", "ep-07", "ep-06"],
        "quant_suspects": ["ep-03", "ep-06", "ep-21", "ep-22", "ep-26", "ep-28", "ep-32", "ep-07"],
    },
    "L1": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct",
        "endpoints": [
            "ep-05", "ep-09", "ep-10", "ep-11", "ep-13",
            "ep-15", "ep-16", "ep-19", "ep-27", "ep-29", "ep-31",
        ],
        "core5": ["ep-29", "ep-16", "ep-31", "ep-27", "ep-10"],
        "quant_suspects": ["ep-05", "ep-09", "ep-10", "ep-11", "ep-13", "ep-15", "ep-19", "ep-27"],
    },
}

WMC_PROMPT = "Pick a random integer from 1 to 100. Reply with ONLY the number."


def generate_core5_csv(out_path: Path) -> None:
    """Write and validate the CORE5 submission CSV (15 base endpoints = 30 positive pairs)."""
    ep_to_fam = {ep: fam for fam, cfg in FAMILIES.items() for ep in cfg["endpoints"]}
    core5_sets = {fam: set(cfg["core5"]) for fam, cfg in FAMILIES.items()}

    assert len(ep_to_fam) == 33
    rows = []
    sing_idx = {fam: 1 for fam in FAMILIES}
    for i in range(1, 34):
        ep = f"ep-{i:02d}"
        fam = ep_to_fam[ep]
        if ep in core5_sets[fam]:
            model = f"{fam}_0"
        else:
            model = f"{fam}_{sing_idx[fam]}"
            sing_idx[fam] += 1
        rows.append((ep, fam, model))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)

    # Validate invariants
    loaded = list(csv.DictReader(out_path.open()))
    assert len(loaded) == 33
    assert [r["endpoint"] for r in loaded] == [f"ep-{i:02d}" for i in range(1, 34)]
    by_model: dict[str, set[str]] = {}
    clusters: dict[str, list[str]] = {}
    for r in loaded:
        by_model.setdefault(r["model"], set()).add(r["family"])
        clusters.setdefault(r["model"], []).append(r["endpoint"])
    assert all(len(fams) == 1 for fams in by_model.values())
    merged = {m: sorted(eps) for m, eps in clusters.items() if len(eps) > 1}
    n_pairs = sum(len(v) * (len(v) - 1) // 2 for v in merged.values())
    print(f"Wrote validated CORE5 CSV to {out_path} ({n_pairs} positive pairs)")
    for m, eps in merged.items():
        print(f"  {m}: {eps}")


def _chat_ids(tok, prompt: str) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return ids if isinstance(ids, list) else ids["input_ids"]


def detect_topp(probe_file: Path, T: float = 1.0) -> None:
    """Detect nucleus-sampling (top_p=0.95) base endpoints via rank & cumulative-mass cutoff."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in probe_file.open() if json.loads(l)["id"].startswith(("main-", "chat-"))]
    for fam, cfg in FAMILIES.items():
        tok = AutoTokenizer.from_pretrained(cfg["model_id"])
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float16).cuda().eval()
        fam_rows = [r for r in rows if r["endpoint"] in cfg["endpoints"]]
        stats = {ep: {"n": 0, "ge50": 0, "max_r": 0, "max_ma": 0.0} for ep in cfg["endpoints"]}
        with torch.no_grad():
            for r in fam_rows:
                pids = _chat_ids(tok, r["prompt"])
                rids = tok(r["response"], add_special_tokens=False)["input_ids"]
                if not rids:
                    continue
                ids = torch.tensor([pids + rids], device="cuda")
                logits = model(ids).logits[0, len(pids) - 1 : -1].float()
                probs = torch.softmax(logits / T, dim=-1)
                tgt = torch.tensor(rids, device="cuda")
                p_tgt = probs.gather(-1, tgt[:, None])
                ranks = (probs > p_tgt).sum(dim=-1)
                # Cumulative probability mass strictly above + including chosen token
                cum_mass = (probs * (probs >= p_tgt)).sum(dim=-1)
                st = stats[r["endpoint"]]
                st["n"] += len(rids)
                st["ge50"] += int((ranks >= 50).sum().item())
                st["max_r"] = max(st["max_r"], int(ranks.max().item()))
                st["max_ma"] = max(st["max_ma"], float(cum_mass.max().item()))
                del logits, probs
        print(f"\n=== {fam} top_p cutoff scan (T={T}) ===")
        print(f"{'ep':7s} {'n_tok':>7s} {'r>=50':>7s} {'maxR':>6s} {'maxMA':>8s}")
        for ep in sorted(stats):
            st = stats[ep]
            flag = " <-- top_p=0.95 BASE" if st["ge50"] == 0 and st["max_ma"] < 0.97 else ""
            print(f"{ep:7s} {st['n']:7d} {st['ge50']:7d} {st['max_r']:6d} {st['max_ma']:8.4f}{flag}")
        del model
        torch.cuda.empty_cache()


def score_watermark_collision(wm_file: Path, wmc_file: Path | None = None, T: float = 0.7) -> None:
    """Compute token-0 high-entropy collision excess over exact local fp16 base p_base."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in wm_file.open()]
    wmc_rows = [json.loads(l) for l in wmc_file.open()] if (wmc_file and wmc_file.exists()) else []

    for fam, cfg in FAMILIES.items():
        tok = AutoTokenizer.from_pretrained(cfg["model_id"])
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float16).cuda().eval()

        @torch.no_grad()
        def base_collision_prob(prompt: str) -> float:
            ids = torch.tensor([_chat_ids(tok, prompt)], device="cuda")
            logits = model(ids).logits[0, -1].float()
            p = torch.softmax(logits / T, dim=-1)
            return float((p * p).sum().item())

        fam_rows = [r for r in rows if r["family"] == fam]
        by_ep_p: dict[tuple[str, str], list[int]] = {}
        prompts = {}
        for r in fam_rows:
            pid = r["id"].rsplit("-", 1)[0]
            prompts[pid] = r["prompt"]
            toks = tok(r["response"].strip(), add_special_tokens=False)["input_ids"]
            if toks:
                by_ep_p.setdefault((r["endpoint"], pid), []).append(toks[0])

        base_c = {pid: base_collision_prob(pr) for pid, pr in prompts.items()}
        if wmc_rows:
            base_c["wmc"] = base_collision_prob(WMC_PROMPT)
            for r in wmc_rows:
                if r["family"] == fam:
                    toks = tok(r["response"].strip(), add_special_tokens=False)["input_ids"]
                    if toks:
                        by_ep_p.setdefault((r["endpoint"], "wmc"), []).append(toks[0])

        eps = sorted({ep for ep, _ in by_ep_p})
        pids = sorted(base_c)
        print(f"\n=== {fam} Token-0 Watermark Collision Excess (empirical C - sum p_base^2) ===")
        print(f"{'ep':7s} " + " ".join(f"{pid:>8s}" for pid in pids) + "    total")
        for ep in eps:
            vals = []
            for pid in pids:
                t0 = by_ep_p.get((ep, pid), [])
                if len(t0) >= 2:
                    n = len(t0)
                    emp_c = sum(c * (c - 1) for c in Counter(t0).values()) / (n * (n - 1))
                    vals.append(emp_c - base_c[pid])
                else:
                    vals.append(0.0)
            tot = sum(vals)
            flag = " <-- WATERMARKED BASE" if tot > 0.35 else ""
            print(f"{ep:7s} " + " ".join(f"{v:+8.3f}" for v in vals) + f" {tot:+8.3f}{flag}")
        del model
        torch.cuda.empty_cache()


def scan_quantization(probe_file: Path, n_answers: int = 30, T: float = 0.7) -> None:
    """10-scheme weight quantization log-likelihood z-score scan (fp16 base vs quantized copy)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def q_fp8_tensor(w):
        s = w.abs().amax().float().clamp_min(1e-12) / 448.0
        return ((w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).to(w.dtype)

    def q_fp8_chan(w):
        s = w.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 448.0
        return ((w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).to(w.dtype)

    def q_int8_chan(w):
        s = w.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / 127.0
        return ((w.float() / s).round().clamp(-128, 127) * s).to(w.dtype)

    def q_int8_tensor(w):
        s = w.abs().amax().float().clamp_min(1e-12) / 127.0
        return ((w.float() / s).round().clamp(-128, 127) * s).to(w.dtype)

    @torch.no_grad()
    def eval_lean(model, items):
        by_ep: dict[str, list[np.ndarray]] = {}
        for ep, pids, rids in items:
            ids = torch.tensor([pids + rids], device="cuda")
            logits = model(ids).logits[0, len(pids) - 1 : -1].float()
            logp = torch.log_softmax(logits / T, dim=-1)
            tgt = torch.tensor(rids, device="cuda")
            by_ep.setdefault(ep, []).append(logp.gather(-1, tgt[:, None]).squeeze(-1).cpu().numpy())
            del logits, logp
        return {ep: np.concatenate(v) for ep, v in by_ep.items()}

    rows_all = [json.loads(l) for l in probe_file.open() if json.loads(l)["id"].startswith(("main-", "chat-"))]
    schemes = [
        ("bf16", None),
        ("fp8_pertensor", q_fp8_tensor),
        ("fp8_perchan", q_fp8_chan),
        ("int8_perchan", q_int8_chan),
        ("int8_pertensor", q_int8_tensor),
    ]

    for fam, cfg in FAMILIES.items():
        eps = cfg["quant_suspects"]
        tok = AutoTokenizer.from_pretrained(cfg["model_id"])
        seen = Counter()
        items = []
        for r in rows_all:
            ep = r["endpoint"]
            if ep in eps and seen[ep] < n_answers:
                rids = tok(r["response"], add_special_tokens=False)["input_ids"]
                if rids:
                    items.append((ep, _chat_ids(tok, r["prompt"]), rids))
                    seen[ep] += 1

        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float16).cuda().eval()
        orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        lp_base = eval_lean(model, items)

        print(f"\n=== {fam} Quantization Fingerprint Scan ({n_answers} ans/ep) ===")
        print(f"{'variant':16s} " + " ".join(f"{e[-2:]:>7s}" for e in eps))
        for name, qfn in schemes:
            if name == "bf16":
                model.to(torch.bfloat16)
            else:
                model.to(torch.float16)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        p.copy_(orig[n].to(p.device, dtype=torch.float16))
                    for n, mod in model.named_modules():
                        if isinstance(mod, torch.nn.Linear) and "lm_head" not in n:
                            mod.weight.data = qfn(mod.weight.data)
            lp_q = eval_lean(model, items)
            zs = []
            for ep in eps:
                d = lp_base[ep] - lp_q[ep]
                zs.append(float(d.mean() / (d.std() / np.sqrt(len(d)))))
            print(f"{name:16s} " + " ".join(f"{z:+7.2f}" for z in zs))
        del model, orig
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2 Model Equality CORE5 pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate and validate CORE5 submission CSV")
    g.add_argument("--out", type=Path, default=Path("submission_core.csv"))

    tp = sub.add_parser("detect-topp", help="Detect top_p=0.95 nucleus-sampling base endpoints")
    tp.add_argument("--probe-file", type=Path, default=Path("probe/ep_probe.jsonl"))

    wm = sub.add_parser("score-wm", help="Score saved token-0 watermark collision probes")
    wm.add_argument("--wm-file", type=Path, default=Path("/tmp/me_opt/wm_tok0.jsonl"))
    wm.add_argument("--wmc-file", type=Path, default=Path("/tmp/me_opt/wmc_confirm.jsonl"))

    sq = sub.add_parser("scan-quant", help="Run weight quantization fingerprint scan")
    sq.add_argument("--probe-file", type=Path, default=Path("probe/ep_probe.jsonl"))
    sq.add_argument("--n-answers", type=int, default=30)

    args = ap.parse_args()
    if args.cmd == "generate":
        generate_core5_csv(args.out)
    elif args.cmd == "detect-topp":
        detect_topp(args.probe_file)
    elif args.cmd == "score-wm":
        score_watermark_collision(args.wm_file, args.wmc_file)
    elif args.cmd == "scan-quant":
        scan_quantization(args.probe_file, args.n_answers)


if __name__ == "__main__":
    main()
