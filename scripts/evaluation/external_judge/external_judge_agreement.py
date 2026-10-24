"""Score ChatGPT replies (chatgpt_batch/reply_XX.txt, pasted verbatim) against the current two-judge labels (answer_key_hidden/answer_key.json).
Prints and writes outputs/chatgpt_agreement.json: coverage, agreement, Cohen's kappa (overall and by stratum), and, on the D vs. B
discordant requests, how many of the two-judge losses/gains ChatGPT reproduces."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, re, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
V = ((OUTPUT_ROOT / "external_judge"))
K = {it["item_id"]: it for it in json.load(open(V / "answer_key_hidden/answer_key.json", encoding="utf-8"))}
G = {}
for f in sorted((V / "chatgpt_batch").glob("reply_*.txt")):
    for m in re.finditer(r"(I[0-9A-F]{6})\s*[,:\-]\s*\**\s*(CORRECT|INCORRECT)", f.read_text(encoding="utf-8").upper()):
        G.setdefault(m.group(1), m.group(2))
def kappa(pairs):
    n = len(pairs)
    if not n: return None
    po = sum(a == b for a, b in pairs) / n; ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[x] * cb[x] for x in ("C", "W")) / n ** 2; return round((po - pe) / (1 - pe), 3) if pe < 1 else None
def summary(ids):
    pr = [("C" if K[i]["label_two_judge"] == "CORRECT" else "W", "C" if G[i] == "CORRECT" else "W") for i in ids if i in G]
    return {"items": len(ids), "answered": len(pr), "agreement_pct": round(100 * sum(a == b for a, b in pr) / len(pr), 1) if pr else None,
            "kappa": kappa(pr), "two_judge_C_chatgpt_W": sum(p == ("C", "W") for p in pr), "two_judge_W_chatgpt_C": sum(p == ("W", "C") for p in pr)}
R = {"overall": summary(list(K)), "by_stratum": {s: summary([i for i in K if K[i]["stratum"] == s]) for s in sorted({k["stratum"] for k in K.values()})}}
# D vs. B direction on the discordant requests (both answers judged)
by_q = collections.defaultdict(dict)
for i, it in K.items():
    if it["stratum"] == "D_vs_B_discordant": by_q[it["q"]][it["policy"]] = i
rep = collections.Counter()
for q, d in by_q.items():
    if len(d) < 2 or d["D"] not in G or d["B"] not in G: rep["incomplete_or_duplicate_answer"] += 1; continue
    tj = "loss" if K[d["B"]]["label_two_judge"] == "CORRECT" else "gain"
    gc = (G[d["B"]] == "CORRECT", G[d["D"]] == "CORRECT")
    gpt = "loss" if gc == (True, False) else "gain" if gc == (False, True) else "no_difference"
    rep[f"two_judge_{tj}__chatgpt_{gpt}"] += 1
R["D_vs_B_discordant_requests"] = dict(rep)
R["missing_item_ids"] = sorted(set(K) - set(G))
(V / "outputs").mkdir(exist_ok=True); json.dump(R, open(V / "outputs/chatgpt_agreement.json", "w"), indent=1); print(json.dumps(R, indent=1))
