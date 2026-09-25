"""Q-16: four-juror query-equivalence audit of SemanticTTL at its
quality-selected operating point (theta=0.60, kappa=1/2).

Replaces the post-hoc theta>=0.60 slice with a real matched measurement on
the selected point's OWN realized-hit population (12,795 hits), using the
published jury protocol verbatim: same system prompt, same four jurors, same
greedy decoding, 2-2 ties resolve to DIFFERENT, band-stratified sample
reweighted to the population.
"""
import json, math, os, pathlib, random, sys, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ART = pathlib.Path(__file__).resolve().parent.parent.parent/"artifacts/q16"
HITS = ART/"semanticttl_selected_hits_t24h.jsonl"
OUTD = ART/"jury_selected"; OUTD.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are evaluating a semantic cache for a question-answering system. "
    "You will be given two questions. Decide whether a single correct answer "
    "would satisfy BOTH questions, so that serving the stored answer for "
    "Question A in response to Question B would be correct.\n\n"
    "Rules:\n"
    "- VERDICT: SAME       if both questions ask for the same fact about the "
    "same entity, even if worded very differently or in different languages.\n"
    "- VERDICT: DIFFERENT  if they ask about different entities (e.g. iPhone 16 "
    "vs iPhone 16 Pro), different attributes of the same entity (e.g. the "
    "trainer vs the jockey), different time references, or different events.\n"
    "- When the questions are merely phrased differently but seek the same "
    "fact, answer SAME.\n"
    "- When in doubt, answer DIFFERENT.\n\n"
    "Respond with exactly one line starting with VERDICT: SAME or "
    "VERDICT: DIFFERENT, followed by a single sentence of reasoning "
    "(max 20 words)."
)
USER = "Question A: {q1}\nQuestion B: {q2}"
JURORS = {"llama3b":"meta-llama/Llama-3.2-3B-Instruct",
          "llama8b":"meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b":"Qwen/Qwen2.5-7B-Instruct",
          "mistral7b":"mistralai/Mistral-7B-Instruct-v0.3"}
BANDS=[(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]
CAP, SEED = 100, 42

def band(s):
    for a,b in BANDS:
        if a<=s<b: return f"{a:.2f}-{b:.2f}"
    return None

pop=[json.loads(l) for l in open(HITS,encoding="utf-8")]
by=collections.defaultdict(list)
for r in pop:
    b=band(r["similarity"])
    if b: by[b].append(r)
print(f"population {len(pop):,} hits; per-band: "
      + ", ".join(f"{b} {len(v):,}" for b,v in sorted(by.items())))

rng=random.Random(SEED); sample=[]
for b,v in sorted(by.items()):
    take=sorted(v,key=lambda r:(r['query_id'],r['matched_query']))
    rng.shuffle(take)
    sel=take[:CAP]
    for r in sel: r=dict(r); r["band"]=b; sample.append(r)
print(f"judged sample: {len(sample)} ({CAP}/band, seed {SEED})")
(OUTD/"sample.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sample))

def verdict(txt):
    t=txt.upper()
    i=t.find("VERDICT:")
    if i>=0: t=t[i:i+40]
    if "DIFFERENT" in t: return "DIFFERENT"
    if "SAME" in t: return "SAME"
    return "DIFFERENT"      # unparseable -> conservative, as published

for jn,mid in JURORS.items():
    out=OUTD/f"{jn}.jsonl"
    if out.exists() and sum(1 for _ in open(out))==len(sample):
        print(f"  {jn}: cached"); continue
    print(f"  {jn}: loading {mid} ...", flush=True)
    tok=AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="cuda:0")
    model.eval()
    res=[]; B=16
    for i in range(0,len(sample),B):
        chunk=sample[i:i+B]; prompts=[]
        for r in chunk:
            u=USER.format(q1=r["matched_query"],q2=r["query"])
            msgs=([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":u}]
                  if "mistral" not in jn else [{"role":"user","content":SYSTEM_PROMPT+"\n\n"+u}])
            prompts.append(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=1024).to(model.device)
        with torch.no_grad():
            gen=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id)
        for r,g,inp in zip(chunk,gen,enc["input_ids"]):
            txt=tok.decode(g[len(inp):],skip_special_tokens=True).strip()
            res.append({"query_id":r["query_id"],"matched_query":r["matched_query"],
                        "band":r["band"],"similarity":r["similarity"],
                        "verdict":verdict(txt),"raw":txt[:200]})
        if i % 160 == 0: print(f"    {i+len(chunk)}/{len(sample)}",flush=True)
    out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in res))
    del model; torch.cuda.empty_cache()
    print(f"  {jn}: done")
print("all jurors complete")
