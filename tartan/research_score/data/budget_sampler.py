from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple



def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _target_count(total: int, fraction: float) -> int:
    return int(math.floor(total * fraction + 0.5))


def nested_window_budgets(
    rows: Sequence[Mapping[str, object]],
    seed: int = 11,
    fractions: Sequence[Tuple[str, float]] = (("1", 0.01), ("10", 0.10), ("100", 1.0)),
) -> Dict[str, object]:
    """Build exact nested budgets inside an already isolated train split."""
    if not rows:
        raise ValueError("train rows must be non-empty")
    ids = [str(r["sample_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id must be unique")
    if any(str(r.get("split")) != "train" for r in rows):
        raise ValueError("all rows must belong to the train split")

    by_stratum: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    by_episode: Dict[str, List[str]] = defaultdict(list)
    meta: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        sid = str(row["sample_id"])
        episode = str(row["episode_id"])
        branch = str(row.get("branch", "unknown"))
        key = (episode, branch)
        by_stratum[key].append(sid)
        by_episode[episode].append(sid)
        meta[sid] = key
    for key in by_stratum:
        by_stratum[key].sort(key=lambda sid: (_rank(seed, sid), sid))
    for episode in by_episode:
        by_episode[episode].sort(key=lambda sid: (_rank(seed, sid), sid))

    ordered_fractions = sorted(fractions, key=lambda x: x[1])
    counts = {tag: _target_count(len(rows), frac) for tag, frac in ordered_fractions}
    episodes = sorted(by_episode)
    if counts[ordered_fractions[0][0]] < len(episodes):
        raise ValueError("smallest budget cannot cover every train episode")

    selected: set[str] = {by_episode[ep][0] for ep in episodes}
    budgets: Dict[str, List[str]] = {}
    stratum_sizes = {key: len(value) for key, value in by_stratum.items()}
    tie_rank = {key: _rank(seed, "|".join(key)) for key in by_stratum}

    def fill(target: int) -> None:
        while len(selected) < target:
            selected_counts = Counter(meta[sid] for sid in selected)
            available = [key for key, queue in by_stratum.items() if any(sid not in selected for sid in queue)]
            if not available:
                raise RuntimeError("budget target exceeds available windows")
            key = min(
                available,
                key=lambda k: (-(target * stratum_sizes[k] / len(rows) - selected_counts[k]), tie_rank[k]),
            )
            selected.add(next(sid for sid in by_stratum[key] if sid not in selected))

    for tag, _ in ordered_fractions:
        fill(counts[tag])
        budgets[tag] = sorted(selected)

    budget_sets = {tag: set(values) for tag, values in budgets.items()}
    memberships = {sid: [tag for tag, _ in ordered_fractions if sid in budget_sets[tag]] for sid in ids}
    summary = {}
    for tag, _ in ordered_fractions:
        chosen = budgets[tag]
        summary[tag] = {
            "windows": len(chosen),
            "episodes": len({meta[sid][0] for sid in chosen}),
            "by_branch": dict(sorted(Counter(meta[sid][1] for sid in chosen).items())),
            "by_episode": dict(sorted(Counter(meta[sid][0] for sid in chosen).items())),
        }
    return {
        "algorithm": "d029_episode_branch_stratified_nested_window_hash_v1",
        "seed": seed,
        "total_windows": len(rows),
        "targets": counts,
        "budgets": budgets,
        "memberships": memberships,
        "summary": summary,
    }


__all__ = ["nested_window_budgets"]
