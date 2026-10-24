"""GPTCache baseline. Uses the official GPTCache package (gptcache==0.1.44 from PyPI, not vendored) with BGE-M3 embeddings and
FreshCache-matched answer admission; only the adapter is part of this repository.

LOCAL ADAPTER code for running the official GPTCache package (gptcache==0.1.44 from PyPI, unmodified) inside the
frozen FreshCache replay harness. The package is not edited; everything here plugs into GPTCache's documented extension points.

  * block_auto_install()   GPTCache calls `pip install` at import time when an optional dependency is missing. That would write outside
                           Baselines, so it is replaced by a function that raises.
  * BGEM3Embedding         embedding_func: returns the frozen pre-computed BGE-M3 vector of the prompt (L2-normalised, float32, dim 1024).
  * FCAdmissionEvaluation  SimilarityEvaluation subclass for the MATCHED variant: score = frozen BGE-M3 cosine of (query, cached question) if
                           the pair passes FreshCache's primary answer-admission rule (cos >= 0.80, content-word Jaccard >= 0.30, answer-type
                           agreement, no strict subset [tiered_cache_model.semantic_equivalent] AND tiered_cache_model._entity_match), else 0. Range (0, 1);
                           with Config(similarity_threshold=0.80) GPTCache's own rank_threshold = 0.80, so only admitted pairs pass.
  * make_cache()           builds a GPTCache Cache with the official sqlite+faiss data manager in a fresh directory (empty cache)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, shutil
import numpy as np


def block_auto_install():
    import gptcache.utils.dependency_control as dc, gptcache.utils as gu
    def _blocked(package, warn=False):
        raise RuntimeError(f"GPTCache tried to pip-install '{package}' (blocked: isolation)")
    dc.prompt_install = _blocked; gu.prompt_install = _blocked


class BGEM3Embedding:
    "Frozen BGE-M3 query embeddings (outputs/l2_verifier/out/emb_queries.npz), keyed by query text."
    dimension = 1024

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True); e = z["emb"].astype(np.float32)
        self.emb = e / np.linalg.norm(e, axis=1, keepdims=True); self.idx = {k: i for i, k in enumerate(z["keys"])}
        self.calls = 0

    def to_embeddings(self, data, **_):
        self.calls += 1
        return self.emb[self.idx[data]]


def make_fc_evaluation(exp):
    from gptcache.similarity_evaluation import SimilarityEvaluation

    class FCAdmissionEvaluation(SimilarityEvaluation):
        def __init__(self):
            self.SIM, self.IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX
            self.n_eval = 0; self.n_ge080 = 0; self.req_ge080 = 0

        def evaluation(self, src_dict, cache_dict, **_):
            q1, q2 = src_dict["question"], cache_dict["question"]
            s = float(self.SIM[self.IDX[q1], self.IDX[q2]]); self.n_eval += 1
            if s >= exp._EQ_SIM_FLOOR: self.n_ge080 += 1; self.req_ge080 += 1
            if s >= exp._EQ_SIM_FLOOR and exp.semantic_equivalent(q1, q2, s) and exp._entity_match(q1, q2):
                return s
            return 0.0

        def range(self):
            return 0.0, 1.0

    return FCAdmissionEvaluation()


def make_cache(data_dir, embedding, evaluation, top_k, similarity_threshold, max_size):
    from gptcache import Cache, Config
    from gptcache.manager import manager_factory
    from gptcache.processor.pre import get_prompt
    from gptcache.processor.post import temperature_softmax
    if os.path.exists(data_dir): shutil.rmtree(data_dir)
    dm = manager_factory("sqlite,faiss", data_dir=data_dir, vector_params={"dimension": embedding.dimension, "top_k": top_k},
                         eviction_params={"max_size": max_size, "eviction": "LRU"})
    c = Cache()
    c.init(pre_embedding_func=get_prompt, embedding_func=embedding.to_embeddings, data_manager=dm, similarity_evaluation=evaluation,
           post_process_messages_func=temperature_softmax, config=Config(similarity_threshold=similarity_threshold, auto_flush=10 ** 9, disable_report=True))
    assert dm.v.count() == 0, "cache not empty at start"
    return c, dm
