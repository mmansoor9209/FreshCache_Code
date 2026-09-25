#!/usr/bin/env python3
"""
The configurable L1 semantic admission condition.

make_gate(...) returns a drop-in replacement for experiment.semantic_equivalent.
It mirrors the published function exactly and adds only what the
pre-registration declares:

  published rule, preserved verbatim:
      1. sim >= sim_floor
      2. content-word Jaccard >= jaccard_floor
      3. answer-type agreement
      4. strict content-token subset rejection
  entity agreement is NOT here: prep2/mixed_engine call experiment._entity_match
  separately and that call is left untouched.

Only the L1 path is affected. L2 and L3 never call semantic_equivalent.
"""
from __future__ import annotations
import pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l1_guards as G   # noqa: E402


def make_gate(exp, sim_floor, jac_floor, answer_type_mode, guards):
    """answer_type_mode: 'as_implemented' | 'required'; guards: bool."""
    assert answer_type_mode in ("as_implemented", "required")

    def semantic_equivalent(q1: str, q2: str, sim: float) -> bool:
        if sim < sim_floor:
            return False
        t1 = exp._eq_content_tokens(q1)
        t2 = exp._eq_content_tokens(q2)
        union = t1 | t2
        if not union:
            return False
        if len(t1 & t2) / len(union) < jac_floor:
            return False
        a1 = exp._eq_answer_type(q1)
        a2 = exp._eq_answer_type(q2)
        if answer_type_mode == "required":
            if a1 is None or a2 is None or a1 != a2:
                return False
        else:
            if a1 is not None and a2 is not None and a1 != a2:
                return False
        # strict-subset guard, verbatim from the published gate
        if t1 != t2 and (t1 <= t2 or t2 <= t1):
            return False
        if guards and not G.structural_guards(q1, q2):
            return False
        return True

    return semantic_equivalent


def published_equivalent(exp, sim, jac):
    """The published gate at its own floors -- used to assert the
    reimplementation is exact before any new cell is trusted."""
    return make_gate(exp, sim, jac, "as_implemented", False)
