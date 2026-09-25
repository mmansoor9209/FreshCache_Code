"""Verify every SHA-256 the manuscript prints, against the shipped files.

The two printed hashes are computed DIFFERENTLY, which is easy to mistake for
a mismatch, so each is checked here with its own rule:

  * MANIFEST.json  -> plain sha256 of the file's bytes.
  * prereg_gamma.json -> a CONTENT hash over the pre-registration body:
        sha256(json.dumps(P, sort_keys=True))  where P is the object with the
        'prereg_sha256' key removed.
    It is deliberately not sha256 of the file, because the field is written
    into the same file it describes and cannot hash itself.

    python scripts/verify_hashes.py          # from the bundle root
"""
import json, hashlib, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ok = True

def report(name, got, want, rule):
    global ok
    good = got == want
    ok &= good
    print(("  PASS " if good else "  FAIL ") + name)
    print(f"        rule      {rule}")
    print(f"        computed  {got}")
    print(f"        printed   {want}")

print("[1] pre-registration content hash (supplement, L2Verify section)")
pg = ROOT/"artifacts/l2_verify/prereg_gamma.json"
d = json.loads(pg.read_text())
P = {k: v for k, v in d.items() if k != "prereg_sha256"}
report("prereg_gamma.json",
       hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest(),
       d["prereg_sha256"],
       "sha256(json.dumps(body, sort_keys=True)), 'prereg_sha256' excluded")
print(f"        note      sha256 of the file as shipped is "
      f"{hashlib.sha256(pg.read_bytes()).hexdigest()}")
print("                  (different by construction; not the printed value)")

print("\n[2] release manifest hash (supplement, release section)")
mp = ROOT.parent/"MANIFEST.json"
if not mp.exists():
    mp = ROOT/"../MANIFEST.json"
stated = (ROOT.parent/"MANIFEST_SHA256.txt")
want = stated.read_text().split()[0] if stated.exists() else None
got = hashlib.sha256(mp.read_bytes()).hexdigest()
if want:
    report("MANIFEST.json", got, want, "plain sha256 of the file's bytes")
else:
    print(f"  computed  {got}")

print("\n[3] the two printed hashes are distinct")
a = d["prereg_sha256"]; b = got
print(("  PASS " if a != b else "  FAIL ") + f"prereg {a[:16]}...  vs  manifest {b[:16]}...")
ok &= (a != b)

print("\n[4] every file listed in MANIFEST.json still hashes to its recorded value")
man = json.loads(mp.read_text())
bad = [f for f, h in man["files"].items()
       if hashlib.sha256((ROOT/f).read_bytes()).hexdigest() != h]
print(("  PASS " if not bad else "  FAIL ") +
      f"{len(man['files']) - len(bad)}/{len(man['files'])} files match")
for f in bad[:10]:
    print("        mismatch:", f)
ok &= not bad

print("\n" + ("ALL HASHES VERIFY" if ok else "SOME HASHES FAILED"))
sys.exit(0 if ok else 1)
