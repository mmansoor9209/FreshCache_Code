#!/usr/bin/env python3
"""REP-01/07/08/09 -- assemble the anonymous release, audit it for identifiers,
and freeze a versioned manifest."""
from __future__ import annotations
import hashlib, json, pathlib, re, datetime

HERE = pathlib.Path(__file__).resolve().parent.parent
REL = HERE / "artifacts" / "release"
REL.mkdir(parents=True, exist_ok=True)
IDENT = re.compile(r"anon|/home/[a-z]+|freshcache|@[a-z0-9.-]+\.(com|edu|org|ac\.kr)"
                   r"|univ|universit|institute|funded by|grant no", re.I)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

man, flagged, n = {}, [], 0
for base in ("artifacts", "scripts"):
    for f in sorted((HERE / base).rglob("*")):
        if not f.is_file() or "release" in f.parts:
            continue
        n += 1
        man[str(f.relative_to(HERE))] = {"sha256": sha(f), "bytes": f.stat().st_size}
        if f.suffix in (".py", ".md", ".json", ".csv", ".tex", ".txt"):
            try:
                head = f.read_text(errors="ignore")[:200_000]
            except Exception:
                continue
            for m in set(IDENT.findall(head)):
                flagged.append((str(f.relative_to(HERE)), str(m)))
json.dump({"version": "freshcache-bench-v1",
           "frozen_utc": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_files": n, "files": man},
          open(REL / "MANIFEST.json", "w"), indent=1)
print(f"  manifest: {n} files")
print(f"  anonymity audit: {len(flagged)} identifier hits in "
      f"{len({f for f,_ in flagged})} files")
by = {}
for f, m in flagged:
    by.setdefault(m.lower(), 0)
    by[m.lower()] += 1
for k, v in sorted(by.items(), key=lambda x: -x[1])[:8]:
    print(f"     {k:<20} {v}")
json.dump({"n_files_flagged": len({f for f, _ in flagged}),
           "patterns": by,
           "files": sorted({f for f, _ in flagged})[:200]},
          open(REL / "ANONYMITY_AUDIT.json", "w"), indent=1)
