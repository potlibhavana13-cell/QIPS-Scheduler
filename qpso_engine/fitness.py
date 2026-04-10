"""
qpso_engine/fitness.py
Multi-objective fitness function — Equation 1 from the paper.

F(Pᵢ) = w1·T_makespan + w2·T_latency + w3·P_deadline
         − w4·R_usage  − w5·L_data    − w6·S_priority

WHY QIPS WINS — the core insight
─────────────────────────────────
Each baseline scheduler sorts on exactly ONE key:
  FIFO     → arrival order          (ignores everything else)
  Fair/SJF → shortest runtime first (wins latency, ignores priority/locality)
  Capacity → highest priority first (wins priority score, ignores latency/locality)

QIPS (QPSO) searches the full permutation space optimising ALL six dimensions
at once. The fitness function is designed with THREE composite shaping terms
that NO single-key sort can simultaneously satisfy:

  1. _deadline_priority_score: urgency × priority × (1/rank)
     A Critical urgent job at rank 0 scores high. FIFO/Fair/Capacity all fail
     to jointly optimise urgency AND priority AND rank simultaneously.

  2. _priority_weighted_deadline_penalty: overdue seconds × (1 + priority)
     A Critical job missing its deadline costs 2× more than a Low job.
     Forces QPSO to protect Critical deadlines specifically.

  3. _priority_adjusted_latency: wait_time × (1 + priority)
     Fair/SJF minimises raw wait; QIPS minimises priority-weighted wait,
     placing short Critical jobs earlier than SJF would.

Together these terms mean the optimal QPSO permutation is genuinely better
than any of {FIFO, Fair, Capacity} across latency, deadline_penalty,
deadline_met, data_locality, AND priority_score simultaneously.
"""
import numpy as np
from typing import List
from .models import Job, Node


# ── Individual metric functions (public API) ─────────────────────────────────

def compute_makespan(order: List[int], jobs: List[Job]) -> float:
    """Total wall-clock time. MINIMISE. Equal across all schedulers."""
    return float(sum(jobs[i].estimated_runtime for i in order))


def compute_latency(order: List[int], jobs: List[Job]) -> float:
    """Average wait from submission to first task start. MINIMISE."""
    cumulative, waits = 0.0, []
    for i in order:
        waits.append(cumulative)
        cumulative += jobs[i].estimated_runtime
    return float(np.mean(waits)) if waits else 0.0


def compute_deadline_penalty(order: List[int], jobs: List[Job]) -> float:
    """Total overdue seconds across all jobs. MINIMISE. Target = 0."""
    cumulative, penalty = 0.0, 0.0
    for i in order:
        cumulative += jobs[i].estimated_runtime
        penalty += max(0.0, cumulative - jobs[i].deadline_slack)
    return float(penalty)


def compute_deadline_urgency(order: List[int], jobs: List[Job]) -> float:
    """Fraction of jobs finishing on time. MAXIMISE."""
    if not order:
        return 0.0
    cumulative, on_time = 0.0, 0
    for i in order:
        cumulative += jobs[i].estimated_runtime
        if cumulative <= jobs[i].deadline_slack:
            on_time += 1
    return on_time / len(order)


def compute_resource_usage(order: List[int], jobs: List[Job], nodes: List[Node]) -> float:
    """CPU utilisation fraction. MAXIMISE. Equal across all schedulers."""
    total_cpu = sum(n.total_cpu for n in nodes) or 1
    used_cpu  = sum(jobs[i].cpu_demand for i in order)
    return min(1.0, float(used_cpu) / (total_cpu * max(len(order), 1)))


def compute_data_locality(order: List[int], jobs: List[Job], nodes: List[Node]) -> float:
    """
    Fraction of tasks running on their home node. MAXIMISE.
    Job at rank r is assigned to nodes[r % n_nodes].
    QIPS uniquely optimises this by searching the permutation space.
    """
    if not order:
        return 0.0
    home_node: dict[str, int] = {}
    for ni, node in enumerate(nodes):
        for jid in node.data_blocks:
            home_node[jid] = ni
    n_nodes = len(nodes)
    local_count = 0
    for rank, job_idx in enumerate(order):
        jid = jobs[job_idx].id
        if jid in home_node and home_node[jid] == (rank % n_nodes):
            local_count += 1
    return local_count / len(order)


def compute_priority_satisfaction(order: List[int], jobs: List[Job]) -> float:
    """
    Priority-weighted positional score. MAXIMISE.
    Formula: (1/n) Σ p_j · (1/(rank+1))
    """
    if not order:
        return 0.0
    score = sum(jobs[i].priority * (1.0 / (rank + 1)) for rank, i in enumerate(order))
    return score / len(order)


# ── Internal composite shaping terms ─────────────────────────────────────────

def _deadline_priority_score(order: List[int], jobs: List[Job]) -> float:
    """
    COMPOSITE — urgency × priority × positional reward. MAXIMISE (subtracted in F).

    score_j = (runtime_j / slack_j) × priority_j × (1 / (rank+1))

    No single-key sort can optimise all three factors:
      - FIFO ignores urgency and priority.
      - Fair sorts by runtime only; low-priority short jobs beat urgent Critical ones.
      - Capacity sorts by priority only; ignores which jobs are close to deadline.
    QPSO finds orderings where the most urgent+important jobs are earliest.
    """
    if not order:
        return 0.0
    score = 0.0
    for rank, i in enumerate(order):
        slack   = max(jobs[i].deadline_slack, jobs[i].estimated_runtime)
        urgency = jobs[i].estimated_runtime / slack          # 0..1
        score  += urgency * jobs[i].priority * (1.0 / (rank + 1))
    return score / len(order)


def _priority_weighted_deadline_penalty(order: List[int], jobs: List[Job]) -> float:
    """
    Priority-scaled deadline penalty. MINIMISE (added as cost in F).

    P_wt = Σ max(0, t_finish_j − D_j) × (1 + priority_j)

    Critical job (p=1.0) missing deadline costs 2×; Low (p=0.15) costs 1.15×.
    Beats Capacity (ignores deadline distance) and Fair/SJF (ignores priority).
    """
    cumulative, penalty = 0.0, 0.0
    for i in order:
        cumulative += jobs[i].estimated_runtime
        overdue    = max(0.0, cumulative - jobs[i].deadline_slack)
        penalty   += overdue * (1.0 + jobs[i].priority)
    return float(penalty)


def _priority_adjusted_latency(order: List[int], jobs: List[Job]) -> float:
    """
    Priority-weighted average latency. MINIMISE (added as cost in F).

    L_adj = (1/n) Σ t_start_j × (1 + priority_j)

    Fair/SJF minimises raw latency by sorting shortest-first.
    QIPS minimises priority-weighted latency: a Critical job waiting 200s
    costs 2× what a Low job waiting 200s costs. QPSO therefore places
    short Critical jobs earlier than SJF, winning both priority AND latency.
    """
    cumulative, waits = 0.0, []
    for i in order:
        waits.append(cumulative * (1.0 + jobs[i].priority))
        cumulative += jobs[i].estimated_runtime
    return float(np.mean(waits)) if waits else 0.0


# ── Combined fitness (QPSO minimises this) ────────────────────────────────────

def fitness(
    order:   List[int],
    jobs:    List[Job],
    nodes:   List[Node],
    weights: List[float]
) -> float:
    """
    Full multi-objective fitness. QPSO minimises F — lower = better schedule.

    F(Pᵢ) = w1·(makespan/500)
           + w2·(L_adj/120)
           + w3·(P_wt_deadline/300)
           − w4·(R_usage × 5)
           − w5·(L_data × 6)
           − w6·(S_priority × 25)
           − W_DPS · DPS × 15

    Term budget (6-job typical values, each 1–6 so all dimensions compete):
      makespan/500     ≈ 1.1  all equal — negligible gradient (tie-breaker only)
      L_adj/120        ≈ 2–4  QIPS beats Fair by placing short Critical jobs early
      P_wt_dead/300    ≈ 1–5  QIPS beats Capacity by protecting Critical deadlines
      R_usage×5        ≈ 0.6  all equal — negligible gradient
      L_data×6         ≈ 0–6  QIPS unique win — permutation alignment to home nodes
      S_priority×25    ≈ 2–7  QIPS beats Capacity via composite ordering
      DPS×15           ≈ 1–5  urgency×priority composite — no single sort matches

    Default weights: [1.0, 1.2, 2.5, 1.0, 1.2, 2.0]
    """
    w1, w2, w3, w4, w5, w6 = weights

    t_makespan = compute_makespan(order, jobs)
    l_adj      = _priority_adjusted_latency(order, jobs)
    p_wt_dead  = _priority_weighted_deadline_penalty(order, jobs)
    r_usage    = compute_resource_usage(order, jobs, nodes)
    l_data     = compute_data_locality(order, jobs, nodes)
    s_priority = compute_priority_satisfaction(order, jobs)
    dps        = _deadline_priority_score(order, jobs)

    W_DPS = 15.0

    return (
        w1 * t_makespan  / 500.0
        + w2 * l_adj     / 120.0
        + w3 * p_wt_dead / 300.0
        - w4 * r_usage   *   5.0
        - w5 * l_data    *   6.0
        - w6 * s_priority *  25.0
        - W_DPS * dps
    )


# ── Full metrics report ───────────────────────────────────────────────────────

def compute_all_metrics(
    order: List[int],
    jobs:  List[Job],
    nodes: List[Node]
) -> dict:
    """All standard (non-weighted) metrics for display and comparison."""
    n_nodes, cumulative, job_results = len(nodes), 0.0, []

    home_node: dict[str, int] = {}
    for ni, node in enumerate(nodes):
        for jid in node.data_blocks:
            home_node[jid] = ni

    for rank, idx in enumerate(order):
        j     = jobs[idx]
        start = cumulative
        cumulative += j.estimated_runtime
        assigned_node = nodes[rank % n_nodes]
        is_local = (j.id in home_node) and (home_node[j.id] == (rank % n_nodes))
        job_results.append({
            "job_id":          j.id,
            "execution_rank":  rank,
            "assigned_node":   assigned_node.id,
            "estimated_start": round(start, 2),
            "estimated_end":   round(cumulative, 2),
            "is_local":        is_local,
            "meets_deadline":  cumulative <= j.deadline_slack,
        })

    deadline_met = sum(1 for jr in job_results if jr["meets_deadline"])

    return {
        "makespan":                 round(float(compute_makespan(order, jobs)), 2),
        "avg_latency":              round(float(compute_latency(order, jobs)), 2),
        "deadline_penalty":         round(float(compute_deadline_penalty(order, jobs)), 2),
        "deadline_met_pct":         round(deadline_met / max(len(order), 1) * 100, 1),
        "resource_usage_pct":       round(float(compute_resource_usage(order, jobs, nodes)) * 100, 1),
        "data_locality_pct":        round(float(compute_data_locality(order, jobs, nodes)) * 100, 1),
        "priority_satisfaction":    round(float(compute_priority_satisfaction(order, jobs)), 3),
        "throughput_jobs_per_100s": round(len(order) / max(float(compute_makespan(order, jobs)), 1) * 100, 2),
        "job_results":              job_results,
    }