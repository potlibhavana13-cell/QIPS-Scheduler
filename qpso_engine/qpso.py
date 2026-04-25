"""
qpso_engine/qpso.py
Quantum-behaved Particle Swarm Optimization — Equations 2–5 from the paper.

QPSO update rule (Eq. 2):
    x_ij(t+1) = p_ij(t) ± β · |m_j(t) − x_ij(t)| · ln(1/u),  u ~ U(0,1)

Baseline schedulers use correct Hadoop definitions:
  FIFO     — strict submission order, no priority/runtime awareness
  Fair     — FIFO within priority tiers (max-min fairness, no SJF)
  Capacity — priority tiers with runtime as tie-breaker (throughput focus)
"""
import numpy as np
from typing import List, Tuple, Optional
from .models import Job, Node
from .fitness import fitness, compute_all_metrics


def _decode(continuous: np.ndarray) -> List[int]:
    """Convert continuous QPSO position to valid job permutation via argsort."""
    return list(np.argsort(continuous))


class QPSOScheduler:
    """
    QPSO-based scheduler. Finds the optimal job execution order.

    Parameters
    ----------
    n_particles : swarm size (default 50 — higher = better but slower)
    max_iter    : maximum iterations (default 200)
    beta        : contraction-expansion coefficient 0.5–1.0 (default 0.75)
    seed        : random seed for reproducibility
    """

    def __init__(
        self,
        n_particles: int = 50,
        max_iter:    int = 200,
        beta:        float = 0.75,
        seed:        Optional[int] = None,
    ):
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.beta        = beta
        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        jobs:    List[Job],
        nodes:   List[Node],
        weights: List[float],
    ) -> Tuple[List[int], float, List[float]]:
        """
        Run QPSO and return the best job execution order found.

        Returns
        -------
        best_order   : list of job indices in optimised execution sequence
        best_fitness : fitness value of the best solution (lower = better)
        history      : fitness value at each iteration (for convergence plot)
        """
        n = len(jobs)
        if n == 0:
            return [], 0.0, []

        def fit(order: List[int]) -> float:
            return fitness(order, jobs, nodes, weights)

        # Initialise swarm with random permutations
        swarm = np.array(
            [np.random.permutation(n).astype(float) for _ in range(self.n_particles)]
        )
        personal_best     = swarm.copy()
        personal_best_fit = np.array([fit(_decode(p)) for p in personal_best])

        g_idx        = int(np.argmin(personal_best_fit))
        global_best  = personal_best[g_idx].copy()
        global_best_fit = personal_best_fit[g_idx]

        history = [float(global_best_fit)]

        for t in range(self.max_iter):
            # Linear beta schedule: starts at beta, ends at beta/2
            beta_t = self.beta * (1.0 - 0.5 * t / self.max_iter)

            # Mean best position across all personal bests
            mbest = personal_best.mean(axis=0)

            for i in range(self.n_particles):
                phi = np.random.uniform(0, 1, n)
                u   = np.clip(np.random.uniform(0, 1, n), 1e-10, 1.0)

                # Attractor: convex combination of personal and global best
                p_ij = phi * personal_best[i] + (1 - phi) * global_best

                # Quantum position update (Equation 2)
                direction = np.where(np.random.rand(n) > 0.5, 1.0, -1.0)
                new_pos   = p_ij + direction * beta_t * np.abs(mbest - swarm[i]) * np.log(1.0 / u)

                swarm[i] = new_pos
                decoded  = _decode(new_pos)
                f        = fit(decoded)

                if f < personal_best_fit[i]:
                    personal_best[i]     = new_pos.copy()
                    personal_best_fit[i] = f
                    if f < global_best_fit:
                        global_best      = new_pos.copy()
                        global_best_fit  = f

            history.append(float(global_best_fit))

        return _decode(global_best), float(global_best_fit), history


# ── Baseline schedulers (correct Hadoop definitions) ─────────────────────────

def fifo_schedule(jobs: List[Job]) -> List[int]:
    """
    FIFO — First-In First-Out.
    Jobs run in strict submission order. No priority or runtime awareness.
    """
    return list(range(len(jobs)))


def fair_schedule(jobs: List[Job]) -> List[int]:
    """
    Hadoop Fair Scheduler approximation.

    The real Fair Scheduler uses max-min fairness between queues — it does
    NOT reorder by runtime (that would be SJF). This approximation groups
    jobs into priority tiers and uses FIFO (submission order) within each
    tier, modelling how fair-share queues drain without runtime-aware reordering.

    Tiers: Critical(1.0) → High(0.75) → Medium(0.4) → Low(0.15)
    """
    tier = {1.0: 0, 0.75: 1, 0.4: 2, 0.15: 3}
    return sorted(
        range(len(jobs)),
        key=lambda i: (tier.get(jobs[i].priority, 2), i)
    )


def capacity_schedule(jobs: List[Job]) -> List[int]:
    """
    Hadoop Capacity Scheduler approximation.

    Partitions cluster into capacity queues per priority tier; within each
    tier, shorter jobs are preferred to maximise throughput — consistent
    with production Capacity Scheduler behaviour. No deadline awareness.
    """
    return sorted(
        range(len(jobs)),
        key=lambda i: (-jobs[i].priority, jobs[i].estimated_runtime)
    )


# ── Run all schedulers for comparison ────────────────────────────────────────

BASELINE_SCHEDULERS = {
    "FIFO":     fifo_schedule,
    "Fair":     fair_schedule,
    "Capacity": capacity_schedule,
}


def run_all_schedulers(
    jobs:        List[Job],
    nodes:       List[Node],
    weights:     List[float],
    qpso_params: Optional[dict] = None,
) -> dict:
    """
    Run QIPS + all baseline schedulers and return metrics for each.
    Used for benchmark comparison tables.
    """
    params    = qpso_params or {}
    scheduler = QPSOScheduler(**params)
    qips_order, qips_fit, qips_hist = scheduler.optimize(jobs, nodes, weights)

    results = {
        "QIPS": {
            "order":           [jobs[i].id for i in qips_order],
            "metrics":         compute_all_metrics(qips_order, jobs, nodes),
            "fitness":         round(qips_fit, 4),
            "fitness_history": [round(v, 4) for v in qips_hist],
        }
    }

    for name, fn in BASELINE_SCHEDULERS.items():
        order = fn(jobs)
        results[name] = {
            "order":           [jobs[i].id for i in order],
            "metrics":         compute_all_metrics(order, jobs, nodes),
            "fitness":         round(fitness(order, jobs, nodes, weights), 4),
            "fitness_history": [],
        }

    return results