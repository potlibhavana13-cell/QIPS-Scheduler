"""
tests/test_qpso.py
Unit and integration tests for the QIPS scheduler.

Run:
  pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qpso_engine import Job, Node, QPSOScheduler, compute_all_metrics
from qpso_engine.fitness import (
    compute_makespan, compute_latency, compute_deadline_penalty,
    compute_resource_usage, compute_data_locality, compute_priority_satisfaction,
    fitness,
)
from qpso_engine.qpso import (
    fifo_schedule, fair_schedule, capacity_schedule, run_all_schedulers
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_jobs():
    return [
        Job(id="J-01", name="WordCount", priority=0.9, estimated_runtime=30,
            cpu_demand=2, mem_demand=1024, deadline_slack=50, input_size_gb=1.0),
        Job(id="J-02", name="PageRank",  priority=0.2, estimated_runtime=120,
            cpu_demand=4, mem_demand=2048, deadline_slack=300, input_size_gb=5.0),
        Job(id="J-03", name="TeraSort",  priority=0.7, estimated_runtime=60,
            cpu_demand=2, mem_demand=1024, deadline_slack=100, input_size_gb=2.0),
    ]

@pytest.fixture
def simple_nodes():
    return [
        Node(id="node-1", total_cpu=8, total_mem=8192, available_cpu=8,
             available_mem=8192, data_blocks=["J-01", "J-03"]),
        Node(id="node-2", total_cpu=8, total_mem=8192, available_cpu=8,
             available_mem=8192, data_blocks=["J-02"]),
        Node(id="node-3", total_cpu=8, total_mem=8192, available_cpu=8,
             available_mem=8192, data_blocks=[]),
    ]

WEIGHTS = [1.0, 1.0, 1.5, 1.0, 1.2, 1.3]


# ── Fitness function tests ─────────────────────────────────────────────────────

class TestFitnessComponents:
    def test_makespan_is_sum_of_runtimes(self, simple_jobs):
        order = [0, 1, 2]
        assert compute_makespan(order, simple_jobs) == pytest.approx(210.0)

    def test_makespan_independent_of_order(self, simple_jobs):
        assert compute_makespan([0,1,2], simple_jobs) == compute_makespan([2,1,0], simple_jobs)

    def test_latency_is_zero_for_first_job(self, simple_jobs):
        order = [1, 0, 2]  # job 1 is first — zero wait
        waits_first = compute_latency([1, 0, 2], simple_jobs)
        # First job has 0 wait; avg will be (0 + 120 + 150) / 3 = 90
        assert waits_first == pytest.approx(90.0)

    def test_deadline_penalty_zero_when_on_time(self, simple_jobs):
        # Jobs with very generous deadlines
        order = [0, 2, 1]  # 30s, 60s, 120s  — cumulative: 30, 90, 210
        # All deadlines: 50, 100, 300  — all met
        assert compute_deadline_penalty(order, simple_jobs) == pytest.approx(0.0)

    def test_deadline_penalty_positive_when_late(self, simple_jobs):
        # J-01 deadline=50, but if J-02 runs first: cumulative at J-01 = 120+60+30=210 > 50
        order = [1, 2, 0]
        penalty = compute_deadline_penalty(order, simple_jobs)
        assert penalty > 0

    def test_resource_usage_in_range(self, simple_jobs, simple_nodes):
        usage = compute_resource_usage([0,1,2], simple_jobs, simple_nodes)
        assert 0.0 <= usage <= 1.0

    def test_data_locality_perfect(self, simple_jobs, simple_nodes):
        # J-01 on node-1 (has J-01 blocks), J-02 on node-2, J-03 on node-1
        order = [0, 1, 2]  # assigned to node-1, node-2, node-3
        # node-1 has J-01 ✓, node-2 has J-02 ✓, node-3 has nothing ✗ → 2/3
        locality = compute_data_locality(order, simple_jobs, simple_nodes)
        assert locality == pytest.approx(2/3, rel=1e-3)

    def test_priority_satisfaction_higher_priority_first(self, simple_jobs):
        # High priority first → better score
        score_good = compute_priority_satisfaction([0, 2, 1], simple_jobs)  # 0.9, 0.7, 0.2
        score_bad  = compute_priority_satisfaction([1, 2, 0], simple_jobs)  # 0.2, 0.7, 0.9
        assert score_good > score_bad

    def test_fitness_lower_is_better(self, simple_jobs, simple_nodes):
        # Optimal order (high priority + tight deadline first) should beat random
        good_order = [0, 2, 1]
        bad_order  = [1, 2, 0]
        f_good = fitness(good_order, simple_jobs, simple_nodes, WEIGHTS)
        f_bad  = fitness(bad_order,  simple_jobs, simple_nodes, WEIGHTS)
        assert f_good < f_bad


# ── QPSO optimizer tests ──────────────────────────────────────────────────────

class TestQPSO:
    def test_returns_valid_permutation(self, simple_jobs, simple_nodes):
        sched = QPSOScheduler(n_particles=10, max_iter=20, seed=42)
        order, fit, history = sched.optimize(simple_jobs, simple_nodes, WEIGHTS)
        assert sorted(order) == [0, 1, 2]
        assert isinstance(fit, float)
        assert len(history) == 21  # max_iter + 1 (initial)

    def test_fitness_decreases_or_stays(self, simple_jobs, simple_nodes):
        sched = QPSOScheduler(n_particles=15, max_iter=50, seed=7)
        _, _, history = sched.optimize(simple_jobs, simple_nodes, WEIGHTS)
        # Global best should never increase
        for i in range(1, len(history)):
            assert history[i] <= history[i-1] + 1e-9

    def test_empty_jobs_returns_empty(self, simple_nodes):
        sched = QPSOScheduler()
        order, fit, history = sched.optimize([], simple_nodes, WEIGHTS)
        assert order == []
        assert history == []

    def test_single_job(self, simple_nodes):
        jobs = [Job(id="J-01", name="Test", priority=0.9, estimated_runtime=10,
                    cpu_demand=1, mem_demand=512, deadline_slack=20, input_size_gb=1.0)]
        sched = QPSOScheduler(n_particles=5, max_iter=10, seed=1)
        order, _, _ = sched.optimize(jobs, simple_nodes, WEIGHTS)
        assert order == [0]

    def test_reproducible_with_seed(self, simple_jobs, simple_nodes):
        s1 = QPSOScheduler(n_particles=10, max_iter=20, seed=99)
        o1, f1, _ = s1.optimize(simple_jobs, simple_nodes, WEIGHTS)
        s2 = QPSOScheduler(n_particles=10, max_iter=20, seed=99)
        o2, f2, _ = s2.optimize(simple_jobs, simple_nodes, WEIGHTS)
        assert o1 == o2
        assert f1 == pytest.approx(f2)


# ── Baseline scheduler tests ──────────────────────────────────────────────────

class TestBaselines:
    def test_fifo_preserves_submission_order(self, simple_jobs):
        assert fifo_schedule(simple_jobs) == [0, 1, 2]

    def test_fair_shortest_first(self, simple_jobs):
        order = fair_schedule(simple_jobs)  # runtimes: 30, 120, 60
        assert order[0] == 0  # J-01 is shortest (30s)

    def test_capacity_highest_priority_first(self, simple_jobs):
        order = capacity_schedule(simple_jobs)  # priorities: 0.9, 0.2, 0.7
        assert simple_jobs[order[0]].priority == 0.9

    def test_run_all_schedulers_returns_all(self, simple_jobs, simple_nodes):
        results = run_all_schedulers(simple_jobs, simple_nodes, WEIGHTS,
                                     qpso_params={"n_particles": 5, "max_iter": 10, "seed": 0})
        assert set(results.keys()) == {"QIPS", "FIFO", "Fair", "Capacity"}
        for name, r in results.items():
            assert "order" in r
            assert "metrics" in r

    def test_qips_beats_fifo_on_priority(self):
        """QIPS should achieve higher priority satisfaction than FIFO in most cases."""
        jobs = [
            Job(id="J-01", name="A", priority=0.05, estimated_runtime=10,
                cpu_demand=1, mem_demand=512, deadline_slack=1000, input_size_gb=1.0),
            Job(id="J-02", name="B", priority=1.0,  estimated_runtime=10,
                cpu_demand=1, mem_demand=512, deadline_slack=1000, input_size_gb=1.0),
            Job(id="J-03", name="C", priority=0.05, estimated_runtime=10,
                cpu_demand=1, mem_demand=512, deadline_slack=1000, input_size_gb=1.0),
            Job(id="J-04", name="D", priority=0.05, estimated_runtime=10,
                cpu_demand=1, mem_demand=512, deadline_slack=1000, input_size_gb=1.0),
        ]
        nodes = [Node(id="n-1", total_cpu=8, total_mem=8192,
                      available_cpu=8, available_mem=8192, data_blocks=[])]
        results = run_all_schedulers(jobs, nodes, [1,1,1,1,1,3],
                                     qpso_params={"n_particles":20,"max_iter":50,"seed":1})
        qips_prio = results["QIPS"]["metrics"]["priority_satisfaction"]
        fifo_prio = results["FIFO"]["metrics"]["priority_satisfaction"]
        assert qips_prio >= fifo_prio


# ── compute_all_metrics tests ─────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_keys_present(self, simple_jobs, simple_nodes):
        m = compute_all_metrics([0,1,2], simple_jobs, simple_nodes)
        required = ["makespan","avg_latency","deadline_penalty",
                    "resource_usage_pct","data_locality_pct","priority_satisfaction",
                    "throughput_jobs_per_100s","job_results"]
        for k in required:
            assert k in m, f"Missing key: {k}"

    def test_job_results_count(self, simple_jobs, simple_nodes):
        m = compute_all_metrics([0,1,2], simple_jobs, simple_nodes)
        assert len(m["job_results"]) == len(simple_jobs)

    def test_job_results_has_required_fields(self, simple_jobs, simple_nodes):
        m = compute_all_metrics([0,1,2], simple_jobs, simple_nodes)
        for jr in m["job_results"]:
            assert "job_id" in jr
            assert "meets_deadline" in jr
            assert "is_local" in jr
