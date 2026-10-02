"""Run the project's validated answer strategies against a running vLLM server.

Methods (same prompts and answer comparison as Phases 7-10):
  single      one answer with the Phase 1 math prompt
  vote        plurality over N independent answers (Phase 9 Option A; N=5 by default)
  self_check  first answer + one blind attempt; if they agree keep it, else a
              judge compares both and writes a final solution (Phase 9/10)

Example:
  python serving/client.py --question "Tom has 3 apples ..." --method self_check \\
      --judge-model phase10-judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402


def chat(base_url: str, model: str, prompt: str, temperature: float, api_key: str | None) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "top_p": 1.0, "max_tokens": 768}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/v1/chat/completions", body, headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", required=True)
    parser.add_argument("--method", choices=("single", "vote", "self_check"), default="vote")
    parser.add_argument("--model", default="original-solver", help="served model or LoRA name")
    parser.add_argument("--judge-model", default=None, help="defaults to --model")
    parser.add_argument("--votes", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    problem = Problem(id="query", domain="math", question=args.question)
    ask = lambda prompt, temperature, model=args.model: chat(args.base_url, model, prompt,
                                                             temperature, args.api_key)
    first = ask(build_prompt(problem), 0.7)
    result = {"method": args.method, "first_answer": first}
    if args.method == "single":
        result["final"] = first
    elif args.method == "vote":
        attempts = [first] + [ask(paired_prompt(problem, "blind_resolve"), 0.7)
                              for _ in range(args.votes - 1)]
        chosen = vote(attempts, list(range(1, len(attempts))) + [0])
        result.update(attempts=attempts, chosen_index=chosen, final=attempts[chosen])
    else:
        blind = ask(paired_prompt(problem, "blind_resolve"), 0.7)
        agree = same(parse(first), parse(blind))
        result.update(blind_attempt=blind, agree=agree)
        if agree:
            result["final"] = first
        else:
            result["final"] = ask(judge_prompt(problem, first, blind), 0.0,
                                  args.judge_model or args.model)
            result["judge_model"] = args.judge_model or args.model
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
