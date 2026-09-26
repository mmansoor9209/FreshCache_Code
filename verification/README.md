# Independent verification audits

Each folder re-derives a group of the paper's numbers from the row-level files in
`artifacts/` and `data/`, with the code lines that produced them quoted verbatim.
Every claim is tagged EXECUTED (recomputed by the audit), RECORD (taken from a
study's saved output) or UNVERIFIED (not checkable from local files).

| Folder | Scope |
|---|---|
| `R13_R14/` | Half-life calibration and domain-volatility filtering |
| `R17/` | Query-equivalence mismatch intervals, including the boundary-aware stratified intervals |
| `R19/` | External benchmarks (EvolvingQA, DailyQA) |
| `R21_stricter_gate/` | Stricter L1 admission study |
| `R21_budget_selection/` | Validation-only budget selection |
| `R16_verifier/` | Optional L2 evidence verifier |
| `R18_R24_timing/` | Replay-routed timing and Monte-Carlo composition |
| `observability/` | Body-observability accounting |
| `R04_reference/` | Reference validity and the 57/107 crosswalk |

The audits were run against the manuscript draft that preceded the final paper.
Where an audit flags a wording or accounting issue, the final paper's text is the
corrected statement; the numbers themselves reproduce as recorded here. Paths in the
reports refer to the original project layout and were sanitised to
`<PROJECT_ROOT>`; the corresponding files in this repository live under
`artifacts/` (see `docs/release/MANIFEST.json`).
