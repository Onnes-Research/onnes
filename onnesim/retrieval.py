"""
retrieval.py — query-conditioned contrastive demonstration selection for the
Diagnostician's few-shot block.

WHY THIS EXISTS
---------------
The paper's enhanced panel prepends a FIXED few-shot block to every query: six
labeled demos chosen by round-robin over the classes (agent_eval.build_few_shot_block).
That block is identical for every scenario — it ignores the telemetry actually being
diagnosed. This module replaces the fixed block with a RETRIEVED one: for each query
window we embed it in the SAME 120-d feature space the ML baseline uses
(ml_baseline.extract_features), find the labeled demos nearest to it, and — crucially —
enforce contrastive coverage of the physically-confusable thermal cluster so the block
always contains the near-miss the model must rule out.

This is a genuine technical lever, not a prompt tweak: demo selection becomes a function
of the query. The hypothesis (tested live in scripts/eval_retrieval.py) is that
query-relevant contrastive demos beat a fixed round-robin block on exactly the confusable
pairs where the zero-shot agent fails.

HONEST SCOPE
------------
The demo bank is drawn from a seed range DISJOINT from both the ML train seeds (0..) and
the evaluation seeds (10,000..), so retrieval never peeks at a test window. Distances are
computed in a standardized feature space (z-scored on the bank) so no single high-magnitude
channel (e.g. a 300 K spare) dominates. Selection is fully deterministic given the bank —
no RNG at query time — so a retrieved block is reproducible.

Nothing here calls an LLM; it only decides WHICH labeled demos to show. The block text
format is identical to agent_eval.build_few_shot_block, so the ONLY variable in the
ablation is which demos are selected, not how they are rendered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np

from . import cryo_engine as CE
from . import ml_baseline as MB
from . import agent as A

# The physically-confusable thermal cluster: these three overlap on temperature and are
# separated only by flow/pressure (see realistic_faults.py). Contrastive coverage means:
# when a query's neighbourhood touches this cluster, show a demo of EACH member so the
# model sees the exact contrast set it must disambiguate.
CONFUSABLE_CLUSTER: tuple[str, ...] = ("helium_leak", "blocked_impedance", "wiring_heat_ingress")

# Class coverage order for building the bank — every class present, confusable ones
# over-represented (they are where classification actually fails).
BANK_ORDER: tuple[str, ...] = (
    "helium_leak", "blocked_impedance", "wiring_heat_ingress", "heat_load_spike",
    "magnet_quench", "normal",
)


@dataclass
class DemoBank:
    """A pool of labeled demonstrations with standardized features for retrieval.

    feats  : [m, d] raw feature vectors (ml_baseline.extract_features)
    zfeats : [m, d] z-scored features (bank mean/std) used for distances
    labels : [m]    ground-truth fault_class per demo
    summaries: [m]  the compact A.summarize_window JSON string per demo (what the LLM sees)
    """
    feats: np.ndarray
    zfeats: np.ndarray
    labels: list[str]
    summaries: list[str]
    mean: np.ndarray
    std: np.ndarray
    seeds: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)


def build_demo_bank(engine_cfg: CE.EngineConfig, per_class: int = 8, demo_seed: int = 500,
                    hours: float = 6.0, dt_min: float = 5.0, sev_scale: float = 0.5) -> DemoBank:
    """Render a labeled demo bank from the disjoint demo-seed range.

    per_class demos are drawn for each of the 6 classes (normal at severity 0). Seeds run
    demo_seed, demo_seed+1, ... and stay well below the eval range (10,000) and above the
    ML train range (0..train_n) for realistic per_class. Features are standardized on the
    bank so nearest-neighbour distances are scale-fair.
    """
    feats: list[np.ndarray] = []
    labels: list[str] = []
    summaries: list[str] = []
    seeds: list[int] = []
    s = demo_seed
    for fc in BANK_ORDER:
        for _ in range(per_class):
            sev = 0.0 if fc == "normal" else 0.7 * sev_scale
            scen = CE.Scenario(fc, sev, 0.4)
            cols = CE.simulate(scen, engine_cfg, hours=hours, dt_min=dt_min, seed=s)
            feats.append(np.asarray(MB.extract_features(cols)[0], dtype=float))
            summaries.append(json.dumps(A.summarize_window(cols)))
            labels.append(fc)
            seeds.append(s)
            s += 1
    F = np.asarray(feats, dtype=float)
    mean = F.mean(axis=0)
    std = F.std(axis=0) + 1e-9
    Z = (F - mean) / std
    return DemoBank(feats=F, zfeats=Z, labels=labels, summaries=summaries,
                    mean=mean, std=std, seeds=seeds)


def _zquery(bank: DemoBank, query_cols: dict) -> np.ndarray:
    q = np.asarray(MB.extract_features(query_cols)[0], dtype=float)
    return (q - bank.mean) / bank.std


def select_demos(bank: DemoBank, query_cols: dict, k: int = 6,
                 contrastive: bool = True) -> list[int]:
    """Return indices into the bank of the k demos to show for this query.

    Base selection: the k nearest bank demos to the query in standardized feature space
    (Euclidean). With contrastive=True we additionally guarantee that every member of the
    confusable thermal cluster is represented by its nearest demo, so the block always
    contains the near-miss contrast (the whole point). Deterministic given the bank."""
    zq = _zquery(bank, query_cols)
    d = np.linalg.norm(bank.zfeats - zq[None, :], axis=1)      # [m]
    order = np.argsort(d, kind="stable")                        # nearest first

    if not contrastive:
        return order[:k].tolist()

    # 1) reserve one slot for the nearest demo of each confusable-cluster class, so the
    #    contrast set is always present regardless of which class dominates the neighbourhood
    picked: list[int] = []
    for fc in CONFUSABLE_CLUSTER:
        cand = [i for i in order if bank.labels[i] == fc]
        if cand:
            picked.append(int(cand[0]))
    # 2) fill the remaining slots with the overall nearest demos not already picked
    for i in order:
        if len(picked) >= k:
            break
        if int(i) not in picked:
            picked.append(int(i))
    # keep at most k, ordered by distance for a stable, readable block
    picked = sorted(set(picked), key=lambda i: d[i])[:k]
    return picked


def format_block(bank: DemoBank, idxs: list[int]) -> str:
    """Render selected demos into the SAME text block format as
    agent_eval.build_few_shot_block, so only the SELECTION differs in the ablation."""
    lines = ["Here are labeled reference examples (telemetry summary -> fault_class). "
             "Use them to distinguish confusable faults that look similar on temperature "
             "but differ on flow/pressure:"]
    for j, i in enumerate(idxs):
        lines.append(f"\nExample {j+1} — fault_class = {bank.labels[i]}:\n{bank.summaries[i]}")
    return "\n".join(lines)


def retrieved_block(bank: DemoBank, query_cols: dict, k: int = 6,
                    contrastive: bool = True) -> str:
    """Convenience: select + format in one call. Returns the few-shot block string."""
    return format_block(bank, select_demos(bank, query_cols, k=k, contrastive=contrastive))
