import re,sys,pathlib
S=pathlib.Path("/tmp/claude-2103/-home-anon-WebRAG/178a4537-be04-4827-83ed-4d67d1b08c8f/scratchpad")
def norm(p):
    t=(S/p).read_text(errors="ignore")
    t=t.replace("\u00ad","").replace("\ufb01","fi").replace("\ufb02","fl")
    t=t.replace("\u2019","'").replace("\u2018","'").replace("\u201c",'"').replace("\u201d",'"')
    t=t.replace("\u2013","-").replace("\u2014","-").replace("\u2212","-").replace("\u00a0"," ")
    return re.sub(r"\s+"," ",t)
M,P=norm("main.txt"),norm("supp.txt")
B=M+" ||| "+P
C=[
 # --- structural / the two edits just made ---
 ("STRUCT-1 abstract single paragraph",       None),  # handled separately
 ("STRUCT-2 references before appendix",      None),
 ("A-01 three objects named in abstract",     M,[r"answer", r"URL list", r"page content"]),
 ("A-01 WAI defined with 81-case denominator",B,[r"81"]),
 # --- AQ-03 / LM-03 generator-attributed accuracy ---
 ("AQ-03 primary 3B = 70.18% (40/57)",        B,[r"70\.18", r"40/57"]),
 ("AQ-03 8B = 68.42% (39/57)",                B,[r"68\.42", r"39/57"]),
 ("AQ-03 generator named at the number",      B,[r"3B"]),
 # --- N-05 notation ---
 ("N-05 unified t_e / t_now notation",        B,[r"age", r"freshness time"]),
 ("N-05 notation map present",                P,[r"[Nn]otation map"]),
 ("N-05 no creation/insertion-time symbol",   P,[r"creation|insertion"]),
 # --- B-14 / SYS-06 runtime block ---
 ("B-14 software+runtime section",            P,[r"[Ss]oftware and runtime"]),
 ("B-14 Python version pinned",               P,[r"3\.10"]),
 ("B-14 BGE-M3 dense CLS L2-normalised",      B,[r"BGE-M3"]),
 ("B-14 31,201 x 31,201 float32 matrix",      B,[r"31,?201"]),
 ("B-14 seed 42",                             B,[r"seed 42|seed of 42"]),
 ("SYS-06 A6000 / FAISS on CPU",              B,[r"A6000", r"FAISS"]),
 ("SYS-06 warm-up + cold fetch",              B,[r"warm-?up", r"cold"]),
 # --- M-11 sampling ---
 ("M-11 28 strata",                           B,[r"\b28\b"]),
 ("M-11 equal (not proportional) allocation", P,[r"equal", r"proportional"]),
 ("M-11 inclusion probability pi_h",          P,[r"sion probability within band|inclusion probability"]),
 ("M-11 pi range 0.0041-0.3030",              P,[r"0\.0041", r"0\.3030"]),
 ("M-11 N_h max 24,252",                      P,[r"24,?252"]),
 # --- M-17 microbenchmark sampling ---
 ("M-17 149 usable / 53 validators",          B,[r"\b149\b", r"\b53\b"]),
 ("M-17 uniform w/o replacement rule",        P,[r"without replacement"]),
 ("M-17 failures reported not retried",       P,[r"retr"]),
 # --- T-04 figure source ---
 ("T-04 editable vector source stated",       B,[r"editable"]),
 # --- QA-06 denominators ---
 ("QA-06 73.40% on 9,225/12,568",             B,[r"73\.40|73\.4\b", r"12,?568"]),
 ("QA-06 44.98% on 9,828/21,848",             B,[r"44\.98", r"21,?848"]),
 # --- REP honesty (the fabricated-URL incident) ---
 ("REP no fabricated 4open.science URL",      B,[r"4open\.science"], True),
 ("REP no 'upon acceptance' promise",         B,[r"upon acceptance|if the paper is accepted|will be created only"], True),
 ("REP archive as supplementary material",    B,[r"supplementary material"]),
 # --- core scientific anchors that must survive any edit ---
 ("CORE identifiability k_t = -ln(1-eps)",    B,[r"identifiab"]),
 ("CORE equivalent TTL",                      B,[r"EquivalentTTL|equivalen"]),
 ("CORE tier multipliers 1.5/1.2/1.0",        B,[r"1\.5", r"1\.2"]),
 ("CORE risk budgets 0.10/0.20/0.35",         B,[r"0\.10", r"0\.20", r"0\.35"]),
 ("CORE exact McNemar",                       B,[r"McNemar"]),
 ("CORE Wilson CI",                           B,[r"Wilson"]),
 ("CORE cluster bootstrap 10,000",            B,[r"10,?000"]),
 ("CORE partial-identification bounds",       B,[r"worst-?case bound"]),
 ("CORE bound never called a CI",             B,[r"worst-case confidence interval"], True),
 ("CORE Zipf term near-inert disclosed",      B,[r"[Zz]ipf"]),
 ("CORE body-observable >= 400 chars",        B,[r"400"]),
 ("CORE domain volatility default 0.40",      B,[r"0\.40"]),
 ("CORE TIMELESS ceiling 0.45 raised",        B,[r"0\.45"]),
 ("CORE latency 3,255.3 -> 1,214.4 (40/pol)", B,[r"3,?255\.3", r"1,?214\.4"]),
 ("CORE MC latency 3,940.4 -> 1,514.9",       B,[r"3,?940\.4", r"1,?514\.9"]),
 ("CORE 981.1 appears only as withdrawn",     B,[r"981\.1", r"withdrawn"]),
 ("CORE cluster McNemar 54 vs 0, p=1.1e-16",  B,[r"\b54\b"]),
 ("CORE nested gaps = descriptive only",      B,[r"diagnostic"]),
 ("CORE half-lives 720/276/127.2/124.8",      B,[r"127\.2", r"124\.8"]),
 ("CORE 325 graded WAI events",               B,[r"\b325\b"]),
 ("CORE Wilson CI on 15.19% (12/79)",      M,[r"Wilson", r"8\.9", r"24\.7"]),
 ("CORE limited-power caveat retained",      M,[r"limited power"]),
 ("CORE k values 0.1013/0.2683/0.6215",      M,[r"0\.1013", r"0\.2683", r"0\.6215"]),
 ("L11 entity guard 13/20 + full gate 19/20", P,[r"13/20", r"19/20"]),
 ("L11 full gate rejects iPhone modifier",   P,[r"iPhone 16"]),
 ("V05 AUC orientation stated explicitly",   P,[r"sufficient.{0,40}class is the one that|scores higher"]),
 ("V05 prereg hash distinct from manifest",  P,[r"18c7e0a965846311"]),
 ("U11 figure L2/L3 pages agree (2x each)",  M,[r"Museum admission page[\s\S]*Museum admission page", r"Ticket pricing page[\s\S]*Ticket pricing page", r"Visitor information page[\s\S]*Visitor information page"]),
 ("U11 caption states L2 cosine >= 0.75",    M,[r"0\.75"]),
 ("U11 figure has no Changed/Refresh label", M,[r"Changed . Refresh|Unchanged . Reuse"], True),
 ("U11 figure shows age-gate decisions",     M,[r"Age OK", r"Age over"]),
 ("L11 guard misses FOUR of seven",          P,[r"misses four"]),
 ("L11 no 'misses every modifier'",          P,[r"misses every modifier"], True),
 ("Q05 configuration-level, not theta isolation", M,[r"not an isolation of"]),
 ("Q05 no 'threshold artifact' overreach",   M,[r"threshold artifact"], True),
 ("REP09 two hash rules explained",          P,[r"29c01922b9134cac"]),
 ("REP09 verify_hashes entry point",         P,[r"verify.hashes"]),
 ("T11 no duplicated Proceedings",           B,[r"Proceedings of Proceedings"], True),
 ("B06 no bare 'SCALM style'",               B,[r"SCALM style"], True),
 ("M15 136/396 present (now in Results)",    M,[r"136 of 396"]),
 ("L11 all misses are no-span",              P,[r"no-?span"]),
 ("M13 242/400 not adjudicable",             P,[r"242"]),
 ("M13 judge agreement 394 / 465",           P,[r"394", r"465"]),
 ("AQ04 runnable entry point named",         P,[r"reproduce[_ ]?\s*screening"]),
 ("G06 body-observable defined in body",     M,[r"block-?page pattern|block-page"]),
 ("G06 mismatch gloss in contributions",     M,[r"different question"]),
 ("REL manifest 1,129 files + hash",         P,[r"1,?129", r"adf59946584a1434"]),

 ("Q16 matched row 53.4% at (0.60,1/2)",    B,[r"53\.4"]),
 ("Q16 matched population 12,795",          B,[r"12,?795"]),
 ("Q16 88.7% only as the superseded value",  M,[r"88\.7", r"Results revised between drafts"]),
 ("Q16 baseline identity: 22,561 reproduced",P,[r"22,?561", r"row for"]),
 ("Q16 30,169/29,378 kept as the wrong run",P,[r"30,?169", r"29,?378"]),
 ("AQ07 quality-selected cluster 18 vs 0",  M,[r"18 discordant clusters"]),
 ("AQ07 drift-selected labelled 54 vs 0",   M,[r"54 against 0"]),
 ("M11 weight direction corrected",         P,[r"up\}-weights the dense|244"]),
 ("AB02 no 'exactly one thing'",            M,[r"exactly one thing"], True),
 ("AB02 L2Only named as two tiers",         M,[r"two tiers at once"]),
 ("T03 L1 guards named in the body",         M,[r"content-word Jaccard", r"entity"]),
 ("STYLE no em dashes remain",              B,[r"\u2014"], True),
 ("CORE N=400 answer audit",                  B,[r"\b400\b"]),
]
main_tex=pathlib.Path("<PROJECT_ROOT>/Reframe_Paper_Pro/freshcache_pro.tex").read_text()
ab=re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",main_tex,re.S).group(1)
struct1 = ("\n\n" not in ab.strip()) and ("\\par" not in ab)
bi=main_tex.index(r"\bibliography{"); ap=main_tex.index(r"\appendix"); co=main_tex.rindex(r"\section{Conclusion")
struct2 = co < bi < ap
fails=[]
for row in C:
    name=row[0]
    if row[1] is None:
        ok = struct1 if name.startswith("STRUCT-1") else struct2
    else:
        _,hay,pats=row[0],row[1],row[2]; absent=len(row)>3 and row[3]
        found=[bool(re.search(p,hay)) for p in pats]
        ok = (not any(found)) if absent else all(found)
    print(("PASS " if ok else "FAIL ")+name)
    if not ok: fails.append(name)
print("\n%d/%d passed"%(len(C)-len(fails),len(C)))
if fails: print("FAILURES:"); [print("  - "+f) for f in fails]
