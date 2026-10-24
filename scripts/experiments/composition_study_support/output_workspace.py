"""Output output_workspace of the shared helpers in composition_study_helpers.py.

Several scripts/experiments share composition_study_helpers.py but keep their own run directory under outputs/<output_workspace>/.
Each script calls use("<output_workspace>") immediately before importing composition_study_helpers; the first call in a process
fixes the output_workspace for the rest of that process."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
NAME = None


def use(name):
    global NAME
    if NAME is None:
        NAME = name
    return NAME


def current():
    if NAME is None:
        raise RuntimeError("output_workspace.use(<name>) must be called before importing composition_study_helpers")
    return NAME
