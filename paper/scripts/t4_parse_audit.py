"""T4: count verifier-wrong s1 answers whose last "Đáp số" line starts with the gold number (format misses)."""
import sys, json, re
sys.path.insert(0, "D:/AGI"); sys.path.insert(0, "D:/AGI/phase1")
from paper.scripts import cpu_checks as cc
from src.core.schema import Problem
from src.data.verifiers.math import MathVerifier
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def first_num_final_line(text):
    i = text.rfind("Đáp số")
    if i < 0: return None
    m = NUM.search(text[i:i + 120])
    return float(m.group().replace(",", "")) if m else None
d = cc.load("gsm8k")
v = MathVerifier()
def audit(name, samples):
    rows = d["rows"]; n = wrong = fixable = cap = 0; ex = []
    for (pid, k), text in samples.items():
        if k != 1: continue
        r = rows[pid]; n += 1
        ok = v.verify(Problem(id=pid, domain="math", question=r["question"], reference_answer=r["reference_answer"]), text).passed
        if ok: continue
        wrong += 1
        x = first_num_final_line(text)
        try: gold = float(str(r["reference_answer"]).replace(",", ""))
        except ValueError: gold = None
        if x is not None and gold is not None and abs(x - gold) < 1e-6:
            fixable += 1; ex.append((pid, text[text.rfind("Đáp số"):][:80].replace("\n", " ")))
    print(name, "n", n, "acc", round(1 - wrong / n, 4), "wrong", wrong, "wrong but first number on final line == gold", fixable,
          "-> corrected acc", round(1 - (wrong - fixable) / n, 4))
    for e in ex[:8]: print("   ", e)
    return {"n": n, "verifier_accuracy": 1 - wrong / n, "format_misses": fixable, "corrected_accuracy": 1 - (wrong - fixable) / n, "examples": ex}
res = {"self_correction_s1": audit("Self_Correction s1", d["samples"])}
t1 = {tuple(r["task"]): r["output"] for r in cc.read_jsonl(__import__("pathlib").Path("D:/AGI/outputs/paper_3090/t1_samples.jsonl"))}
res["instruct_s1"] = audit("Instruct s1", t1)
json.dump(res, open("D:/AGI/paper/results/t4_parse_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
