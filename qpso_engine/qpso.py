"""
qpso_engine/qpso.py
Quantum-behaved Particle Swarm Optimization — Equations 2–5 from the paper.

QPSO position update (Eq. 2):
    x_ij(t+1) = p_ij(t) ± β · |m_j(t) − x_ij(t)| · ln(1/u),  u ~ U(0,1)

Unlike standard PSO, QPSO has no velocity vector — each particle moves
probabilistically around the attractor point p_ij, guided by the mean-best
position m_j of the swarm.  This eliminates three velocity hyperparameters
(ω, c1, c2) and improves convergence in high-dimensional discrete spaces.
"""
import numpy as np
from typing import List, Tuple
from .models import Job, Node
from .fitness import fitness, compute_all_metrics


def _random_permutation(n: int) -> np.ndarray:
    return np.random.permutation(n).astype(float)


def _decode(continuous: np.ndarray) -> List[int]:
    """Convert continuous QPSO position to a valid job permutation via argsort."""
    return list(np.argsort(continuous))


class QPSOScheduler:
    """
    QPSO-based scheduler that finds the optimal job execution order.

    Parameters
    ----------
    n_particles  : swarm size
    max_iter     : maximum number of iterations
    beta         : contraction-expansion coefficient (controls exploration vs exploitation)
                   Typical range: 0.5 – 1.0.  Paper uses ~0.75.
    seed         : random seed for reproducibility
    """

    def __init__(
        self,
        n_particles: int = 30,
        max_iter: int = 100,
        beta: float = 0.75,
        seed: int | None = None
    ):
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.beta        = beta
        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        jobs: List[Job],
        nodes: List[Node],
        weights: List[float]
    ) -> Tuple[List[int], float, List[float]]:
        """
        Run QPSO and return the best job execution order found.

        Returns
        -------
        best_order   : list of job indices in optimized execution sequence
        best_fitness : fitness value of the best solution
        history      : fitness value at each iteration (for convergence plot)
        """
        n = len(jobs)
        if n == 0:
            return [], 0.0, []

        def fit(order_indices: List[int]) -> float:
            return fitness(order_indices, jobs, nodes, weights)

        # ── Initialization ────────────────────────────────────────────────────
        swarm = np.array([_random_permutation(n) for _ in range(self.n_particles)])

        personal_best     = swarm.copy()
        personal_best_fit = np.array([fit(_decode(p)) for p in personal_best])

        g_idx        = int(np.argmin(personal_best_fit))
        global_best  = personal_best[g_idx].copy()
        global_best_fit = personal_best_fit[g_idx]

        history = [float(global_best_fit)]

        # ── Main QPSO loop ────────────────────────────────────────────────────
        for t in range(self.max_iter):

            # Linear beta schedule: start at beta, end at beta/2
            beta_t = self.beta * (1.0 - 0.5 * t / self.max_iter)

            # Mean best position (mbest) — average of all personal bests
            mbest = personal_best.mean(axis=0)

            for i in range(self.n_particles):
                phi = np.random.uniform(0, 1, n)
                u   = np.clip(np.random.uniform(0, 1, n), 1e-10, 1.0)

                # Attractor: convex combination of personal and global best (Eq. 2)
                p_ij = phi * personal_best[i] + (1 - phi) * global_best

                # Quantum random direction ± with equal probability
                direction = np.where(np.random.rand(n) > 0.5, 1.0, -1.0)

                # QPSO position update
                new_pos = p_ij + direction * beta_t * np.abs(mbest - swarm[i]) * np.log(1.0 / u)

                # Decode to permutation
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

        best_order = _decode(global_best)
        return best_order, float(global_best_fit), history


# ── Baseline schedulers (for comparison) ─────────────────────────────────────

def fifo_schedule(jobs: List[Job]) -> List[int]:
    """First-In First-Out: submission order (index order)."""
    return list(range(len(jobs)))


def fair_schedule(jobs: List[Job]) -> List[int]:
    """Fair: shortest job first (approximates fair share)."""
    return sorted(range(len(jobs)), key=lambda i: jobs[i].estimated_runtime)


def capacity_schedule(jobs: List[Job]) -> List[int]:
    """Capacity: sort by priority then runtime (simulates capacity queue tiers)."""
    return sorted(range(len(jobs)), key=lambda i: (-jobs[i].priority, jobs[i].estimated_runtime))


BASELINE_SCHEDULERS = {
    "FIFO":     fifo_schedule,
    "Fair":     fair_schedule,
    "Capacity": capacity_schedule,
}


def run_all_schedulers(
    jobs: List[Job],
    nodes: List[Node],
    weights: List[float],
    qpso_params: dict | None = None
) -> dict:
    """
    Run QIPS + all baseline schedulers and return metrics for each.
    Useful for benchmark comparisons.
    """
    params = qpso_params or {}
    scheduler = QPSOScheduler(**params)
    qips_order, qips_fit, qips_hist = scheduler.optimize(jobs, nodes, weights)

    results = {
        "QIPS": {
            "order":         [jobs[i].id for i in qips_order],
            "metrics":       compute_all_metrics(qips_order, jobs, nodes),
            "fitness":       round(qips_fit, 4),
            "fitness_history": [round(v, 4) for v in qips_hist],
        }
    }

    for name, fn in BASELINE_SCHEDULERS.items():
        order = fn(jobs)
        results[name] = {
            "order":   [jobs[i].id for i in order],
            "metrics": compute_all_metrics(order, jobs, nodes),
            "fitness": round(fitness(order, jobs, nodes, weights), 4),
            "fitness_history": [],
        }

    return results
