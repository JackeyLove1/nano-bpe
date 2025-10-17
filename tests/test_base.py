'''
pytest -q -m performance tests/test_base.py
'''
import logging
import time
import random
import pytest  # type: ignore

from nanobpe.base import get_stats, get_stats_torch
from nanobpe.base import merge, merge_torch


def _generate_ids(length: int, vocab_size: int, seed: int = 42):
    random.seed(seed)
    return [random.randrange(vocab_size) for _ in range(length)]


def _time_function(fn, *args, runs: int = 3, **kwargs) -> float:
    # warmup
    fn(*args, **kwargs)
    start = time.perf_counter()
    for _ in range(runs):
        fn(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) / runs


def test_get_stats_correctness_matches_torch(monkeypatch):
    # Force torch path to run on CPU for stability across environments
    try:
        import torch  # type: ignore  # noqa: F401
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    except Exception:
        pass

    ids = _generate_ids(length=5000, vocab_size=1024)

    py_counts = get_stats(ids)
    torch_counts = get_stats_torch(ids)

    assert py_counts == torch_counts


'''
GPU times contain data host2device and device2host
get_stats (pure Python): 6.390499 s (avg)
get_stats_torch (Torch GPU): 44.655956 s (avg)
'''
@pytest.mark.performance
def test_get_stats_performance_report(monkeypatch, capsys):
    # Force torch path to run on CPU for stability across environments
    try:
        import torch  # type: ignore  # noqa: F401
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    except Exception:
        pass

    ids = _generate_ids(length=10_000_000, vocab_size=128000)

    t_py = _time_function(get_stats, ids, runs=3)
    t_torch = _time_function(get_stats_torch, ids, runs=3)

    logging.info(f"get_stats (pure Python): {t_py:.6f} s (avg)")
    logging.info(f"get_stats_torch (Torch GPU): {t_torch:.6f} s (avg)")


def test_merge_functional_examples():
    # empty and single element
    assert merge([], (1, 2), 99) == []
    assert merge([7], (1, 2), 99) == [7]

    # simple pair
    assert merge([1, 2], (1, 2), 9) == [9]
    assert merge([1, 2, 3], (1, 2), 9) == [9, 3]
    assert merge([3, 1, 2], (1, 2), 9) == [3, 9]

    # multiple occurrences
    assert merge([1, 2, 1, 2], (1, 2), 4) == [4, 4]
    assert merge([1, 2, 3, 1, 2], (1, 2), 4) == [4, 3, 4]

    # overlapping pairs should be non-overlapping/greedy left-to-right
    assert merge([1, 1, 1], (1, 1), 8) == [8, 1]
    assert merge([1, 1, 1, 1], (1, 1), 8) == [8, 8]


def _ensure_pair_present(ids, pair, every: int = 10):
    # insert the pair periodically to ensure presence for testing
    a, b = pair
    out = []
    for i, v in enumerate(ids):
        out.append(v)
        if i % every == 0:
            out.extend([a, b])
    return out


def test_merge_torch_matches_merge_random(monkeypatch):
    # Force torch path to run on CPU for stability across environments
    try:
        import torch  # type: ignore  # noqa: F401
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    except Exception:
        pass

    random.seed(123)
    vocab_size = 100
    pair = (7, 13)
    idx_new = 255

    # random base ids, then ensure the pair appears multiple times
    ids = [random.randrange(vocab_size) for _ in range(1000)]
    ids = _ensure_pair_present(ids, pair, every=7)

    expected = merge(ids, pair, idx_new)
    got = merge_torch(ids, pair, idx_new)
    assert got == expected

    # also when the pair does not exist
    pair_absent = (98, 99)
    assert merge_torch(ids, pair_absent, idx_new) == merge(ids, pair_absent, idx_new)


'''
GPU times contain data host2device and device2host
merge (pure Python): 3.089811 s (avg)
merge_torch (Torch): 1.158884 s (avg)
'''
@pytest.mark.performance
def test_merge_vs_merge_torch_performance_report(monkeypatch, capsys):
    # Force torch path to run on CPU for stability across environments
    try:
        import torch  # type: ignore  # noqa: F401
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    except Exception:
        pass

    # Construct a long list with many opportunities to merge
    vocab_size = 512
    base_ids = _generate_ids(length=10_000_000, vocab_size=vocab_size, seed=7)
    ids = _ensure_pair_present(base_ids, pair=(21, 42), every=5)

    t_py = _time_function(merge, ids, (21, 42), 70000, runs=3)
    t_torch = _time_function(merge_torch, ids, (21, 42), 70000, runs=3)

    logging.info(f"merge (pure Python): {t_py:.6f} s (avg)")
    logging.info(f"merge_torch (Torch): {t_torch:.6f} s (avg)")


