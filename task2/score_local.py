"""Teacher-forced scoring of endpoint answers under a local Hugging Face model.

For every answer in a jsonl file ({"endpoint", "id", "repeat", "prompt", "response"}) the prompt is wrapped with the
model's chat template (no system message, as the API does, unless --system-prompts gives a recovered one), the response is re-tokenised and fed through the model.
Per response token we keep:
  lp[T]   log-prob of the observed token under softmax(logits / T) for every T in TEMPS
  ent[T]  entropy of softmax(logits / T); if the local model is the true sampler at T, E[lp + ent] = 0
  mass_above[T]  probability mass of tokens ranked above the observed one; PIT u = mass_above + U * exp(lp)
                 is uniform if the local model at T is the true sampler (rank-based test)
  rank    rank of the observed token (0 = argmax)
  margin  logit gap between the top-1 and top-2 token (small = near-tie, where quantisation flips greedy choices)
Results go to a pickle: {(endpoint, id, repeat): dict(lp=[n, len(TEMPS)], rank=[n], margin=[n], tokens=[n])}.

Usage:
  python score_local.py --model HuggingFaceTB/SmolLM2-360M-Instruct --answers probe/ref_probe.jsonl \
                        --out probe/scores_360m.pkl [--quant fp8|int8|int4] [--dtype float32] [--limit-endpoints ref-01,ref-02]
"""
import argparse
import json
import pickle
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@torch.no_grad()
def simulate_fp8(model: torch.nn.Module) -> int:
    """Round-trips every Linear weight (except the LM head) through float8_e4m3fn with a per-tensor scale, as
    weight-only fp8 quantisation does. Returns the number of tensors changed."""
    changed = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            w = module.weight.data
            scale = w.abs().max().float() / torch.finfo(torch.float8_e4m3fn).max
            q = (w.float() / scale).to(torch.float8_e4m3fn).float() * scale
            module.weight.data = q.to(w.dtype)
            changed += 1
    return changed


@torch.no_grad()
def simulate_int(model: torch.nn.Module, bits: int, group_size: int | None) -> int:
    """Round-trips every Linear weight (except the LM head) through symmetric absmax integer quantisation:
    per output channel (group_size None, like int8 W8A16) or per group of input columns (like int4 GPTQ/AWQ g128)."""
    qmax = 2 ** (bits - 1) - 1
    changed = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            w = module.weight.data
            wf = w.float()
            shape = wf.shape
            if group_size:
                wf = wf.reshape(shape[0], -1, group_size)
            scale = wf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
            q = (wf / scale).round().clamp(-qmax - 1, qmax) * scale
            module.weight.data = q.reshape(shape).to(w.dtype)
            changed += 1
    return changed


NF4_LEVELS = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                           -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                           0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                           0.7229568362236023, 1.0])


@torch.no_grad()
def simulate_nf4(model: torch.nn.Module, block: int = 64) -> int:
    """bitsandbytes-style NF4: per block of 64 weights, absmax scale, round to the nearest of 16 normal-float levels."""
    changed = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            w = module.weight.data
            flat = w.float().reshape(-1, block)
            scale = flat.abs().amax(-1, keepdim=True).clamp_min(1e-12)
            levels = NF4_LEVELS.to(w.device)
            idx = ((flat / scale)[..., None] - levels).abs().argmin(-1)
            module.weight.data = (levels[idx] * scale).reshape(w.shape).to(w.dtype)
            changed += 1
    return changed


QUANT = {
    "nf4": simulate_nf4,
    "fp8": simulate_fp8,
    "int8": lambda m: simulate_int(m, 8, None),
    "int4": lambda m: simulate_int(m, 4, 128),
}


def load_answers(path: Path, endpoints: set | None) -> list:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if endpoints is None or r["endpoint"] in endpoints]


@torch.no_grad()
def score_one(model, tokenizer, prompt: str, response: str, device: str, system: str | None = None,
              light: bool = False) -> dict:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if not isinstance(prompt_ids, list):        # newer transformers may return a BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not response_ids:
        return None
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    logits = model(ids).logits[0, len(prompt_ids) - 1:-1].float()           # predicts each response token
    target = torch.tensor(response_ids, device=device)
    chosen = logits.gather(-1, target[:, None])
    above = logits > chosen                                                   # tokens ranked above the observed one
    # One temperature at a time keeps peak memory at a few [n, V] tensors (shared login GPUs run out otherwise).
    lp, ent, mass_above = [], [], []
    for t in TEMPS:
        logp = torch.log_softmax(logits / t, -1)                              # [n, V]
        lp.append(logp.gather(-1, target[:, None]).squeeze(-1))
        if not light:
            probs = logp.exp()
            ent.append(-(probs * logp).sum(-1))                               # entropy = -E[lp] if model true
            mass_above.append((probs * above).sum(-1))                        # for the PIT / rank test
            del probs
        del logp
    lp = torch.stack(lp, 1)                                                   # [n, T]
    if light:                                   # quantised copies only need lp for the likelihood ratio
        return {"lp": lp.cpu().numpy(), "tokens": response_ids}
    ent, mass_above = torch.stack(ent, 1), torch.stack(mass_above, 1)
    rank = (logits > chosen).sum(-1)
    top2 = logits.topk(2, -1).values
    return {"lp": lp.cpu().numpy(), "ent": ent.cpu().numpy(), "mass_above": mass_above.cpu().numpy(), "rank": rank.cpu().numpy(), "margin": (top2[:, 0] - top2[:, 1]).cpu().numpy(),
            "tokens": response_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp8", action="store_true", help="shorthand for --quant fp8")
    ap.add_argument("--quant", choices=QUANT, default=None, help="score with a weight-only quantised copy")
    ap.add_argument("--dtype", default="float32", choices=DTYPES)
    ap.add_argument("--limit-endpoints", default=None, help="comma-separated endpoint ids")
    ap.add_argument("--light", action="store_true", help="store only lp (enough for likelihood ratios)")
    ap.add_argument("--system-prompts", type=Path, default=None,
                    help="json {endpoint: recovered system prompt}; those endpoints are scored with it as system message")
    args = ap.parse_args()
    system_prompts = json.loads(args.system_prompts.read_text()) if args.system_prompts else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype]).to(device).eval()
    quant = "fp8" if args.fp8 else args.quant
    if quant:
        print(f"{quant} round-trip applied to {QUANT[quant](model)} Linear weights")

    endpoints = set(args.limit_endpoints.split(",")) if args.limit_endpoints else None
    rows = load_answers(args.answers, endpoints)
    results = {}
    for i, r in enumerate(rows):
        scored = score_one(model, tokenizer, r["prompt"], r["response"], device, system_prompts.get(r["endpoint"]),
                           light=args.light)
        if scored is not None:
            results[(r["endpoint"], r["id"], r["repeat"])] = scored
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump({"model": args.model, "fp8": quant or False, "dtype": args.dtype, "temps": TEMPS,
                     "results": results}, f)
    print(f"Saved {len(results)} scored answers to {args.out}")


if __name__ == "__main__":
    main()
