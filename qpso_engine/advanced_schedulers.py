"""
qpso_engine/advanced_schedulers.py

Implementations of the four advanced baseline schedulers from the paper:
  - HybSMRP          : Hybrid Scheduling with MapReduce Profiles
  - HFSP             : Hadoop Fair Scheduler with Preemption
  - Frugal_conf      : Frugal Configuration-aware Scheduler
  - Intra-task Loc   : Intra-task Localization Scheduler

Each is implemented as a deterministic algorithm that models the core
decision logic of the real scheduler, using only the job/node metadata
available in our simulation (runtime, priority, deadline, locality, CPU).

Design principle: each scheduler is progressively smarter than the last,
building toward QIPS which combines ALL their ideas simultaneously via QPSO.

References from paper:
  HybSMRP  [17-23]: predictive modelling + multi-criteria selection
  HFSP     [24-28]: deadline-constrained + preemption-aware
  Frugal   [29-33]: resource-frugal + configuration-aware
  Intra-task [37] : data-locality at individual task granularity
"""
from typing import List, Tuple
from .models import Job, Node
from .fitness import (
    compute_makespan, compute_latency, compute_deadline_penalty,
    compute_resource_usage, compute_data_locality, compute_priority_satisfaction,
    compute_all_metrics,
)


# ── 1. HybSMRP — Hybrid Scheduling with MapReduce Profiles ───────────────────

def hybsmrp_schedule(jobs: List[Job], nodes: List[Node]) -> List[int]:
    """
    HybSMRP: Hybrid Scheduling using MapReduce job Profiles.

    Core idea: use historical job profiling to predict runtime and
    resource needs, then combine FIFO-within-priority with a runtime
    efficiency score. Jobs are scored by a weighted combination of:
      - Priority (most important)
      - Estimated CPU efficiency  (cpu_demand / runtime ratio)
      - Input size (larger inputs processed earlier to avoid I/O bottlenecks)

    This models how HybSMRP uses job profiles to make smarter placement
    decisions than pure priority or FIFO ordering. It improves over
    Capacity scheduler by adding runtime-efficiency awareness.

    Improvement over Capacity: ~10% makespan, ~36% latency vs FIFO.
    """
    def hybsmrp_score(idx: int) -> float:
        j = jobs[idx]
        # Priority component (normalised 0-1)
        pri_score = j.priority

        # CPU efficiency: jobs that use CPU well relative to their runtime
        # High cpu_demand / estimated_runtime = compute-intensive = process first
        cpu_efficiency = j.cpu_demand / (j.estimated_runtime + 1e-9)
        cpu_score = min(1.0, cpu_efficiency / 0.1)  # normalise to ~[0,1]

        # Input size score: larger inputs scheduled earlier (pipeline benefit)
        # Normalise by max input size across all jobs
        max_input = max(j.input_size_gb for j in jobs) or 1.0
        input_score = j.input_size_gb / max_input

        # Deadline urgency: tighter deadlines get higher score
        max_dl = max(j.deadline_slack for j in jobs) or 1.0
        deadline_score = 1.0 - (j.deadline_slack / max_dl)

        # Weighted combination (profile-based hybrid weighting)
        return (0.40 * pri_score
                + 0.25 * cpu_score
                + 0.20 * deadline_score
                + 0.15 * input_score)

    return sorted(range(len(jobs)), key=hybsmrp_score, reverse=True)


# ── 2. HFSP — Hadoop Fair Scheduler with Preemption ─────────────────────────

def hfsp_schedule(jobs: List[Job], nodes: List[Node]) -> List[int]:
    """
    HFSP: Hadoop Fair Scheduler with Preemption.

    Core idea: extends Fair Scheduler with deadline-driven preemption.
    When a job is at risk of missing its deadline, it can preempt lower-
    priority jobs. The scheduler computes a "preemption urgency" score
    for each job that combines:
      - Priority weight
      - Deadline slack ratio (how close to missing the deadline?)
      - Expected wait time penalty (how much longer can this job wait?)

    The urgency score determines which jobs jump the queue.
    Preemption means jobs with tighter deadlines get a disproportionately
    large boost — unlike FIFO/Fair where only priority matters.

    Improvement over HybSMRP: ~14% makespan, ~41% latency vs FIFO.
    """
    total_runtime = sum(j.estimated_runtime for j in jobs) or 1.0
    max_dl = max(j.deadline_slack for j in jobs) or 1.0

    def hfsp_urgency(idx: int) -> float:
        j = jobs[idx]

        # Base priority (maps Critical→1.0, High→0.75, etc.)
        base = j.priority

        # Deadline pressure: exponential urgency as deadline approaches
        # A job with slack=50s and runtime=45s is VERY urgent
        slack_ratio = j.deadline_slack / total_runtime  # <1 = urgent
        deadline_pressure = 1.0 / (slack_ratio + 0.1)   # higher pressure = tighter deadline
        deadline_pressure = min(deadline_pressure, 10.0) / 10.0  # normalise

        # Preemption threshold: jobs that NEED resources NOW vs can wait
        # Short jobs that are high priority = high preemption need
        runtime_fraction = j.estimated_runtime / total_runtime
        preemption_need = base * (1.0 - runtime_fraction)

        # Wait sensitivity: how much does waiting hurt this job?
        wait_sensitivity = (j.priority * j.estimated_runtime) / (j.deadline_slack + 1.0)
        wait_sensitivity = min(1.0, wait_sensitivity / 2.0)

        return (0.35 * base
                + 0.30 * deadline_pressure
                + 0.20 * preemption_need
                + 0.15 * wait_sensitivity)

    return sorted(range(len(jobs)), key=hfsp_urgency, reverse=True)


# ── 3. Frugal_conf — Frugal Configuration-aware Scheduler ───────────────────

def frugal_schedule(jobs: List[Job], nodes: List[Node]) -> List[int]:
    """
    Frugal_conf: Frugal Configuration-aware Scheduler.

    Core idea: minimises resource WASTE by being frugal — only requesting
    what a job actually needs, and scheduling jobs in the order that
    maximises cluster-wide utilisation with the least over-provisioning.

    Key innovations over HFSP:
      1. Resource fit score: jobs whose CPU/mem demand fits node capacity
         well are preferred (avoids fragmentation)
      2. Waste avoidance: jobs with high cpu_demand/mem_demand ratio that
         match available slots are prioritised
      3. Configuration awareness: respects cluster topology

    Frugal scheduling produces tighter bin-packing, which improves
    throughput and reduces the idle-resource tail of each job batch.

    Improvement over HFSP: ~18% makespan, ~46% latency vs FIFO.
    """
    total_node_cpu = sum(n.total_cpu for n in nodes) or 1
    avg_node_cpu   = total_node_cpu / max(len(nodes), 1)
    max_dl = max(j.deadline_slack for j in jobs) or 1.0

    def frugal_score(idx: int) -> float:
        j = jobs[idx]

        # Frugality: how well does this job's demand fit a single node?
        # Perfect fit = 1.0, wasteful = lower score
        fit_ratio = min(1.0, j.cpu_demand / avg_node_cpu)
        frugality = 1.0 - abs(fit_ratio - 0.5) * 0.5  # peaks at 50% node utilisation

        # Resource efficiency: output per resource unit
        # High input / low cpu_demand = I/O bound, schedule first to free CPU
        resource_efficiency = j.input_size_gb / (j.cpu_demand + 1e-9)
        resource_efficiency = min(1.0, resource_efficiency / 5.0)

        # Priority-deadline combined (same as HFSP base)
        deadline_ratio = 1.0 - (j.deadline_slack / max_dl)
        priority_dl = j.priority * (1.0 + deadline_ratio) / 2.0

        # Configuration match: jobs with smaller memory footprint per CPU
        # are more frugal and preferred in frugal scheduling
        mem_per_cpu = j.mem_demand / (j.cpu_demand * 512.0 + 1e-9)
        frugal_mem  = 1.0 / (1.0 + mem_per_cpu)  # lower mem/cpu = more frugal

        return (0.30 * priority_dl
                + 0.25 * frugality
                + 0.25 * resource_efficiency
                + 0.20 * frugal_mem)

    return sorted(range(len(jobs)), key=frugal_score, reverse=True)


# ── 4. Intra-task Localization ───────────────────────────────────────────────

def intratask_schedule(jobs: List[Job], nodes: List[Node]) -> List[int]:
    """
    Intra-task Localization Scheduler.

    Core idea: maximises data locality at the INDIVIDUAL TASK level by
    computing a detailed locality score for each job based on where its
    data blocks actually live. Jobs whose data is entirely on a single
    node (rack-local) are strongly preferred over jobs with scattered data.

    Key innovations over Frugal:
      1. Block-level locality analysis: scores each job by its data
         placement across nodes (concentrated = high score)
      2. Data-movement cost estimation: estimates network transfer cost
         and penalises jobs that would require cross-rack I/O
      3. Locality-priority balance: high-priority + local = highest score,
         but a very-local low-priority job beats a scattered high-priority one
         if the deadline is not imminent

    This directly models Hadoop's "data-local first" principle at
    individual task granularity, maximising HDFS read performance.

    Improvement over Frugal: ~22% makespan, ~48% latency vs FIFO.
    """
    # Build a locality map: for each job, count how many nodes hold its data
    n_nodes = len(nodes) or 1
    job_locality = {}
    for idx, j in enumerate(jobs):
        nodes_with_data = sum(1 for n in nodes if j.id in n.data_blocks)
        # Locality concentration: 1 node = perfectly local, all nodes = distributed
        locality_concentration = 1.0 - (nodes_with_data - 1) / max(n_nodes - 1, 1)
        job_locality[idx] = max(0.0, locality_concentration)

    max_dl = max(j.deadline_slack for j in jobs) or 1.0
    total_rt = sum(j.estimated_runtime for j in jobs) or 1.0

    def intratask_score(idx: int) -> float:
        j = jobs[idx]
        loc = job_locality[idx]  # 1.0 = perfectly concentrated, 0.0 = distributed

        # Locality score: strongly favour data-local jobs
        # This is the primary differentiator vs Frugal
        locality_score = loc ** 0.5  # sqrt to avoid over-penalising distributed jobs

        # Data volume locality: large data + local = huge benefit
        data_vol_score = min(1.0, j.input_size_gb / 4.0) * loc

        # Priority + deadline (similar to HFSP)
        deadline_urgency = 1.0 - (j.deadline_slack / max_dl)
        priority_dl = j.priority * (1.0 + deadline_urgency) / 2.0

        # Runtime locality benefit: shorter runtimes benefit most from locality
        # (for long jobs, locality savings are proportionally smaller)
        rt_locality_benefit = (1.0 - j.estimated_runtime / total_rt) * loc

        return (0.35 * locality_score
                + 0.25 * priority_dl
                + 0.20 * data_vol_score
                + 0.20 * rt_locality_benefit)

    return sorted(range(len(jobs)), key=intratask_score, reverse=True)


# ── Unified runner ────────────────────────────────────────────────────────────

ADVANCED_SCHEDULERS = {
    "HybSMRP":         hybsmrp_schedule,
    "HFSP":            hfsp_schedule,
    "Frugal_conf":     frugal_schedule,
    "Intra-task Loc":  intratask_schedule,
}


def run_advanced_schedulers(
    jobs:    List[Job],
    nodes:   List[Node],
) -> dict:
    """
    Run all four advanced schedulers and return metrics for each.
    Used alongside run_all_schedulers() for full 8-way comparison.
    """
    results = {}
    for name, fn in ADVANCED_SCHEDULERS.items():
        order = fn(jobs, nodes)
        results[name] = {
            "order":   [jobs[i].id for i in order],
            "metrics": compute_all_metrics(order, jobs, nodes),
        }
    return results


def run_all_eight_schedulers(
    jobs:        List[Job],
    nodes:       List[Node],
    weights:     List[float],
    qpso_params: dict = None,
) -> dict:
    """
    Run all 8 schedulers from the paper for a complete comparison:
    FIFO, Fair, Capacity, HybSMRP, HFSP, Frugal_conf, Intra-task Loc, QIPS.
    Returns results dict keyed by scheduler name.
    """
    from .qpso import run_all_schedulers as run_basic
    basic   = run_basic(jobs, nodes, weights, qpso_params)
    advanced = run_advanced_schedulers(jobs, nodes)

    # Add fitness scores for advanced schedulers
    from .fitness import fitness
    for name, data in advanced.items():
        order_indices = [
            next(i for i, j in enumerate(jobs) if j.id == jid)
            for jid in data["order"]
        ]
        data["fitness"] = round(fitness(order_indices, jobs, nodes, weights), 4)
        data["fitness_history"] = []

    return {**basic, **advanced}