"""Token-excess vectors: where does an endpoint's text deviate from the local base model, in vocabulary space?

For each answer, at the endpoint's fitted temperature T* (taken from a score_local.py pickle):
    excess[v] = (# times token v was emitted) - sum over positions of p_{T*}(v | prefix)
The base model explains the text -> excess is pure noise. A LoRA (logits + s * delta) pushes mass in a consistent
direction, so excess vectors of the same adapter at different strengths correlate.

Saves per endpoint the excess and expected-count vectors summed over two halves of the prompts (even / odd prompt
number), shape (2, V), so a direction estimated on one half can be tested on the other.

Usage: python excess.py --model HuggingFaceTB/SmolLM2-360M-Instruct --answers probe/ref_probe.jsonl \
                        --scores probe/s360_base.pkl --out probe/excess_360.pkl
"""
import argparse
import collections
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_local import load_answers


def fitted_temperatures(scores: dict) -> dict:
    temps = np.array(scores["temps"])
    per_ep = collections.defaultdict(list)
    for (ep, _, _), r in scores["results"].items():
        per_ep[ep].append(r["lp"] + r["ent"])
    return {ep: float(np.interp(0, np.concatenate(v).mean(0), temps)) for ep, v in per_ep.items()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True, help="score_local.py pickle used to fit T* per endpoint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-endpoints", default=None, help="comma-separated endpoint ids")
    args = ap.parse_args()

    with args.scores.open("rb") as f:
        t_star = fitted_temperatures(pickle.load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    vocab = len(tokenizer)                      # real tokens; the embedding matrix may be padded beyond this

    endpoints = set(args.limit_endpoints.split(",")) if args.limit_endpoints else None
    # Sums per endpoint over two halves of the prompts (even / odd prompt number), so that a direction estimated on
    # one half can be tested on the other without sharing prompts. Memory stays at 2 x V per endpoint.
    excess = collections.defaultdict(lambda: np.zeros((2, vocab), np.float64))
    expected_sum = collections.defaultdict(lambda: np.zeros((2, vocab), np.float64))
    n_answers = collections.Counter()
    for r in load_answers(args.answers, endpoints):
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True)
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids["input_ids"]
        response_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        if not response_ids:
            continue
        ids = torch.tensor([prompt_ids + response_ids], device=device)
        logits = model(ids).logits[0, len(prompt_ids) - 1:-1].float()
        expected = torch.softmax(logits / t_star[r["endpoint"]], -1).sum(0)[:vocab]
        observed = torch.bincount(torch.tensor(response_ids, device=device), minlength=vocab).float()[:vocab]
        half = int(r["id"].rsplit("-", 1)[1]) % 2
        excess[r["endpoint"]][half] += (observed - expected).cpu().numpy()
        expected_sum[r["endpoint"]][half] += expected.cpu().numpy()
        n_answers[r["endpoint"]] += 1
    with args.out.open("wb") as f:
        pickle.dump({"t_star": t_star, "excess": dict(excess), "expected": dict(expected_sum),
                     "n_answers": dict(n_answers)}, f)
    print(f"Saved excess halves for {len(excess)} endpoints to {args.out}; T*: "
          + ", ".join(f"{ep}={t_star[ep]:.2f}" for ep in sorted(excess)))


if __name__ == "__main__":
    main()
