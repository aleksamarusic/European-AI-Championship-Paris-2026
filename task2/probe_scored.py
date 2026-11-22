"""Step 2: family identification on the scored endpoints ep-01..33 (spends real budget: 10 answers per endpoint).

3 identity prompts (who made you) + 7 long open-ended chat prompts (~250 tokens each; the 4 DET prompts were validated
on the reference endpoints). The chat answers are reused later for the teacher-forced trap tests.
Answers go to probe/ep_probe.jsonl. Restart-safe: answers already in the file are skipped.

Usage: python probe_scored.py
"""
from pathlib import Path

from probe_refs import DET_PROMPTS
from task_template import query, make_session, get_budget

OUT = Path("probe/ep_probe.jsonl")

IDENTITY_PROMPTS = [
    "What is your name and who created you?",
    "Which company or organization trained you, and what is your exact model name and version?",
    "Describe yourself: what model are you, roughly how many parameters do you have, and what is your knowledge cutoff?",
]
CHAT_PROMPTS = DET_PROMPTS + [
    "Write a detailed essay about the history of the bicycle.",
    "Explain step by step how a computer processor executes a program.",
    "Write a recipe for a hearty vegetable soup with detailed instructions.",
]


def main():
    budget = get_budget(make_session())
    endpoints = [ep for ep in budget if ep.startswith("ep-")]
    print(f"{len(endpoints)} scored endpoints, min budget left {min(budget[e] for e in endpoints)}")
    OUT.parent.mkdir(exist_ok=True)
    prompts = [(f"id-{i}", p) for i, p in enumerate(IDENTITY_PROMPTS)] + \
              [(f"chat-{i}", p) for i, p in enumerate(CHAT_PROMPTS)]
    query(prompts, output=OUT, endpoints=endpoints, repeats=1)


if __name__ == "__main__":
    main()
