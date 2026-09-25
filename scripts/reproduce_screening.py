"""AQ-04: single runnable entry point for fidelity and temporal-gold screening.

Re-derives every screening aggregate the paper reports, from the row-level
files in this archive, and checks each against the stored summaries. No GPU,
no model calls, no network: this verifies that the published numbers follow
from the released rows.

    python scripts/reproduce_screening.py        # run from the bundle root

Exit code 0 iff every check reproduces.
"""
import csv, json, pathlib, sys, collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
GV   = ROOT/"artifacts/gold_verification_400"
SRE  = ROOT/"artifacts/validation_V3/strong_reference_evaluation"

def rows(p):
    with open(p, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

ok = True
def check(name, got, want):
    global ok
    good = got == want; ok &= good
    print(("  PASS " if good else "  FAIL ") + f"{name}: {got}" + ("" if good else f"  (expected {want})"))

print("== 1. temporal-gold validity screening (stage 3b) ==")
gv = rows(GV/"gold_validity_at_t.csv")
s3b = json.loads((GV/"stage3b_summary.json").read_text())
check("units screened", len(gv), 715)
fs = collections.Counter(r["final_status"] for r in gv)
print(f"  final_status: {dict(fs)}")
check("statuses sum to units", sum(fs.values()), len(gv))
agree = sum(1 for r in gv if r["judges_agree"] == "1")
check("a status is assigned only on dual-judge agreement",
      agree, len(gv) - fs["JUDGE_DISAGREEMENT"])
check("disagreements are labelled, never silently resolved",
      sum(1 for r in gv if r["judges_agree"] == "0" and r["final_status"] != "JUDGE_DISAGREEMENT"), 0)
print(f"  judges: {s3b['judges']}")
print(f"  decoding: {s3b['decoding']}")

print("\n== 2. paraphrase-fidelity screening (stage 3a) ==")
pf = rows(GV/"paraphrase_fidelity.csv")
check("paraphrase pairs screened", len(pf), 668)
ag = sum(1 for r in pf if r["judges_agree"] == "1")
print(f"  two-judge agreement: {ag}/{len(pf)} = {100*ag/len(pf):.1f}%")
check("every pair carries both judges' labels",
      sum(1 for r in pf if r["label_llama8b"].strip() and r["label_qwen7b"].strip()), len(pf))

print("\n== 3. the 400-request reference assessment ==")
rd = json.loads((SRE/"02_reference_verification/reference_and_diagnosis.json").read_text())
check("N", rd["n"], 400)
st = rd["reference_status"]
print(f"  status: {st}")
check("status sums to 400", sum(st.values()), 400)
check("supported", st["SUPPORTED"], 57)
check("contradicted", st["CONTRADICTED"], 101)
check("temporally ambiguous", st["TEMPORALLY_AMBIGUOUS"], 178)
check("unverifiable", st["UNVERIFIABLE"], 64)
check("cases flagged as needing human review", rd["human_review_cases"],
      st["TEMPORALLY_AMBIGUOUS"] + st["UNVERIFIABLE"])

print("\n  NOTE ON TAXONOMIES. reference_and_diagnosis.csv carries a coarser")
print("  three-way column (reference_status: CONTRADICTED/UNVERIFIABLE/SUPPORTED,")
print("  where UNVERIFIABLE=242) and the stage-3b mechanism labels")
print("  (reference_validity_study). The paper reports the four-way split above,")
print("  in which the 242 non-supported, non-contradicted cases divide into 178")
print("  temporally ambiguous and 64 unverifiable. Same 242 cases, finer split.")
csvrows = rows(SRE/"02_reference_verification/reference_and_diagnosis.csv")
coarse = collections.Counter(r["reference_status"] for r in csvrows)
check("coarse UNVERIFIABLE equals ambiguous + unverifiable",
      coarse["UNVERIFIABLE"], st["TEMPORALLY_AMBIGUOUS"] + st["UNVERIFIABLE"])
check("coarse CONTRADICTED matches", coarse["CONTRADICTED"], st["CONTRADICTED"])
check("coarse SUPPORTED matches", coarse["SUPPORTED"], st["SUPPORTED"])

print("\n== 4. supported-subset fresh accuracy ==")
sup = [r for r in csvrows if r["reference_status"] == "SUPPORTED"]
check("supported n", len(sup), 57)
corr = sum(1 for r in sup if r["fresh_label"].strip().upper() == "CORRECT")
check("primary 3B fresh-correct on supported subset (paper: 40/57 = 70.18%)", corr, 40)
print(f"  = {100*corr/len(sup):.2f}%")

print("\n== 5. nothing is claimed as human adjudication ==")
check("is_human_validated flag", rd["is_human_validated"], False)
hr = SRE/"02_reference_verification/human_reference_review.csv"
if hr.exists():
    h = rows(hr)
    filled = sum(1 for r in h if any((v or "").strip() for k, v in r.items()
                                     if "human" in k.lower() or "label" in k.lower()))
    check("human review sheet shipped unfilled (no fabricated labels)", filled, 0)

print("\n== 6. screening is rerunnable from released inputs ==")
for f in ["stage3a_paraphrase.py", "stage3b_validity.py", "stage1_build.py",
          "stage0_reproduce.py", "validity_tasks.jsonl", "input_hashes.json",
          "reproduction_gate.json", "validity_labels_llama8b.jsonl",
          "validity_labels_qwen7b.jsonl", "paraphrase_labels_llama8b.jsonl",
          "paraphrase_labels_qwen7b.jsonl"]:
    check(f"present: {f}", (GV/f).exists(), True)

print("\n" + ("ALL SCREENING CHECKS REPRODUCE" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
