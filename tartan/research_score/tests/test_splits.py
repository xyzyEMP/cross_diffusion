import pytest

from tartan.research_score.data.budget_sampler import nested_window_budgets
from tartan.research_score.data.core import assert_no_split_leak, group_split, nested_budgets


def test_group_split_and_nested():
    splits = group_split([f"p{i}" for i in range(24)])
    assert set(splits.values()) == {"train", "val", "test"}
    budgets = nested_budgets([x for x, value in splits.items() if value == "train"], 11)
    assert set(budgets["1"]) <= set(budgets["10"]) <= set(budgets["100"])


def test_d029_exact_nested_episode_coverage_and_order_invariance():
    rows = [
        {"sample_id": f"e{episode}:{i:03d}", "episode_id": f"e{episode}",
         "branch": "moving_planning" if i % 3 else "stop_or_short", "split": "train"}
        for episode in range(16) for i in range(128)
    ]
    result = nested_window_budgets(rows, seed=11)
    assert result["targets"] == {"1": 20, "10": 205, "100": 2048}
    one, ten, full = (set(result["budgets"][tag]) for tag in ("1", "10", "100"))
    assert one < ten < full
    assert len({sid.split(":")[0] for sid in one}) == 16
    assert result["budgets"] == nested_window_budgets(list(reversed(rows)), seed=11)["budgets"]


def test_d029_rejects_cross_split_rows():
    rows = [
        {"sample_id": "a", "episode_id": "e1", "branch": "moving_planning", "split": "train"},
        {"sample_id": "b", "episode_id": "e2", "branch": "moving_planning", "split": "val"},
    ]
    with pytest.raises(ValueError, match="train split"):
        nested_window_budgets(rows)


def test_no_leak():
    assert_no_split_leak([{"map_id": "a", "trajectory_id": "t", "split": "train"}])
