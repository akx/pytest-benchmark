"""
Tests for the ``--benchmark-precision`` stopping rule.

Most tests call ``BenchmarkFixture._run_until_precise`` directly with a fake runner
so the statistics are deterministic.
"""

import random

import pytest

from pytest_benchmark.fixture import BenchmarkFixture
from pytest_benchmark.utils import t_quantile

# the earliest the batch means rule can stop: one check per batch, and the target has to
# hold on that many checks in a row
MIN_STOP = BenchmarkFixture.PRECISION_MIN_BATCHES + BenchmarkFixture.PRECISION_CONFIRMATIONS - 1


class FakeStats:
    def __init__(self):
        self.data = []

    def update(self, duration):
        self.data.append(duration)


def make_runner(values):
    it = iter(values)
    return lambda loops_range: next(it)


def configure(benchmark, precision, min_rounds=5, confidence=0.99):
    benchmark._mode = 'benchmark(...)'  # mark the fixture as used since we bypass __call__
    benchmark._min_rounds = min_rounds
    benchmark._precision = precision
    benchmark._confidence = confidence
    return benchmark


def test_early_zero_variance_stops(benchmark):
    configure(benchmark, precision=0.01, min_rounds=5)
    stats = FakeStats()
    benchmark._run_until_precise(make_runner([1.0] * 100), range(1), stats, 50)
    assert len(stats.data) == MIN_STOP
    assert benchmark._precision_result['converged'] is True


def test_min_rounds_respected(benchmark):
    configure(benchmark, precision=0.01, min_rounds=MIN_STOP + 17)
    stats = FakeStats()
    benchmark._run_until_precise(make_runner([1.0] * 200), range(1), stats, 200)
    assert len(stats.data) >= MIN_STOP + 17


def test_never_precise_runs_to_cap(benchmark):
    # make sure that noisy samples never reach the tight margin
    configure(benchmark, precision=0.0001, min_rounds=2)
    rng = random.Random(0)
    stats = FakeStats()
    benchmark._run_until_precise(lambda loops_range: rng.lognormvariate(0, 0.5), range(1), stats, 200)
    assert len(stats.data) == 200


def test_alternating_rounds_average_out(benchmark):
    configure(benchmark, precision=0.001, min_rounds=2)
    stats = FakeStats()
    benchmark._run_until_precise(make_runner([1.0, 5.0] * 500), range(1), stats, 1000)
    assert benchmark._precision_result['converged'] is True
    assert len(stats.data) < 1000


def test_reports_target_unreachable(benchmark):
    # running out of rounds is not the same as being precise, so it shouldn't look like it
    configure(benchmark, precision=0.0001, min_rounds=2)
    warnings = []
    benchmark._warner = warnings.append
    rng = random.Random(0)
    stats = FakeStats()
    benchmark._run_until_precise(lambda loops_range: rng.lognormvariate(0, 0.5), range(1), stats, 100)
    result = benchmark._precision_result
    assert result['converged'] is False
    assert result['achieved'] > result['target']
    assert 'short of the requested' in str(warnings[0])


def testprecise_stops_early(benchmark):
    configure(benchmark, precision=0.05, min_rounds=3)
    stats = FakeStats()
    samples = [0.99, 1.01] * 500
    benchmark._run_until_precise(make_runner(samples), range(1), stats, 1000)
    assert MIN_STOP <= len(stats.data) < 1000


def test_long_run_batches_grow(benchmark):
    # batches are merged pairwise once there are too many of them, so a long run keeps a
    # bounded number of increasingly large batches instead of every round it ever took
    configure(benchmark, precision=0.0005, min_rounds=5)
    rng = random.Random(0)
    stats = FakeStats()
    benchmark._run_until_precise(lambda loops_range: rng.lognormvariate(0, 0.2), range(1), stats, 5000)
    assert benchmark._precision_result['batch_size'] > 1


def test_quiet_stretch_doesnt_end(benchmark):
    configure(benchmark, precision=0.01, min_rounds=5)
    quiet = [1.0] * (MIN_STOP - 1)
    noisy = [3.0, 1.0] * 200
    stats = FakeStats()
    benchmark._run_until_precise(make_runner(quiet + noisy), range(1), stats, 300)
    assert len(stats.data) > MIN_STOP


def test_zero_durations_never_converge(benchmark):
    configure(benchmark, precision=0.01, min_rounds=2)
    warnings = []
    benchmark._warner = warnings.append
    stats = FakeStats()
    benchmark._run_until_precise(make_runner([0.0] * 100), range(1), stats, 30)
    assert len(stats.data) == 30
    result = benchmark._precision_result
    assert result['converged'] is False
    assert result['achieved'] is None
    assert 'unknown' in str(warnings[0])


def test_interval_uses_student_t():
    # small batch counts need the wider t quantile, not the normal one
    assert t_quantile(0.9975, 19) == pytest.approx(3.174, abs=0.001)
    assert t_quantile(0.995, 30) == pytest.approx(2.750, abs=0.001)
    assert t_quantile(0.9975, 100_000) == pytest.approx(2.807, abs=0.001)


def test_t_quantile_rejects_empty_samples():
    # a single batch leaves no degrees of freedom, so there's no interval to compute
    with pytest.raises(ValueError, match='df must be positive'):
        t_quantile(0.99, 0)


@pytest.mark.benchmark(precision=0.05, max_time=2.0, min_rounds=3)
def test_precision_marker_end_to_end(benchmark):
    # Full path: marker options through the fixture to the adaptive loop.
    benchmark(lambda: sum(range(50)))
    assert benchmark.stats.stats.rounds >= 3
    assert benchmark.stats.precision['target'] == 0.05
    saved = benchmark.stats.as_dict()
    assert saved['precision'] == benchmark.stats.precision
    assert saved['options']['precision'] == 0.05


def test_no_precision_when_fixed_rounds(benchmark):
    # without --benchmark-precision there is no margin to report, and the key stays out
    benchmark(lambda: sum(range(50)))
    assert benchmark.stats.precision is None
    assert 'precision' not in benchmark.stats.as_dict()
