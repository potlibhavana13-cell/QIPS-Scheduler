"""
simulator/app.py
Flask web server for the QIPS scheduler simulator.
Supports all 8 schedulers from the paper.
Compatible with both local development and Vercel/serverless deployment.

Run locally:
  python simulator/app.py
  Open http://localhost:5000
"""

import sys
import os
import random
import numpy as np

# Ensure root path is available for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from flask import Flask, render_template, jsonify, request

from qpso_engine import (
    QPSOScheduler, Job, Node, compute_all_metrics,
    run_all_schedulers,
    hybsmrp_schedule, hfsp_schedule, frugal_schedule, intratask_schedule,
    fifo_schedule, fair_schedule, capacity_schedule,
)

# ✅ Fix template path for deployment
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

JOB_NAMES = [
    "WordCount", "PageRank", "TeraSort", "HiveQuery", "SparkMLlib",
    "HBase Scan", "Pig Latin", "MapReduce ETL", "Flume Ingest", "Sqoop Import",
]

PRIORITY_TIERS = [
    ("Critical", 1.0), ("High", 0.75), ("Medium", 0.4), ("Low", 0.15)
]

SCHEDULER_MAP = {
    "FIFO":           lambda jobs, nodes, w, p: fifo_schedule(jobs),
    "Fair":           lambda jobs, nodes, w, p: fair_schedule(jobs),
    "Capacity":       lambda jobs, nodes, w, p: capacity_schedule(jobs),
    "HybSMRP":        lambda jobs, nodes, w, p: hybsmrp_schedule(jobs, nodes),
    "HFSP":           lambda jobs, nodes, w, p: hfsp_schedule(jobs, nodes),
    "Frugal_conf":    lambda jobs, nodes, w, p: frugal_schedule(jobs, nodes),
    "Intra-task Loc": lambda jobs, nodes, w, p: intratask_schedule(jobs, nodes),
    "QIPS":           None,  # handled separately via QPSOScheduler
}

SCHEDULER_ORDER = [
    "FIFO", "Fair", "Capacity",
    "HybSMRP", "HFSP", "Frugal_conf", "Intra-task Loc",
    "QIPS",
]


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def sanitize(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    return obj


# ─────────────────────────────────────────────────────────────
# Workload Generator
# ─────────────────────────────────────────────────────────────

def generate_workload(n_jobs: int = 6, n_nodes: int = 3, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = [
        Node(
            id=f"Node-{i+1}",
            total_cpu=8,
            total_mem=8192,
            available_cpu=8,
            available_mem=8192,
            data_blocks=[],
        )
        for i in range(n_nodes)
    ]

    jobs = []
    deadline_factors = [1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for i in range(n_jobs):
        name = JOB_NAMES[i % len(JOB_NAMES)]
        pri_label, pri_val = random.choice(PRIORITY_TIERS)
        local_node_idx = random.randint(0, n_nodes - 1)

        runtime = random.randint(20, 180)
        deadline = int(runtime * random.choice(deadline_factors))

        j = Job(
            id=f"J-{str(i+1).zfill(2)}",
            name=name,
            priority=pri_val,
            estimated_runtime=float(runtime),
            cpu_demand=random.randint(1, 6),
            mem_demand=random.choice([512, 1024, 2048, 4096]),
            deadline_slack=float(deadline),
            input_size_gb=round(random.uniform(0.5, 8.0), 1),
            local_node_id=nodes[local_node_idx].id,
        )

        jobs.append(j)
        nodes[local_node_idx].data_blocks.append(j.id)

    return jobs, nodes


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/workload", methods=["POST"])
def new_workload():
    body = request.get_json(force=True)

    jobs, nodes = generate_workload(
        n_jobs=int(body.get("n_jobs", 6)),
        n_nodes=int(body.get("n_nodes", 3)),
        seed=body.get("seed"),
    )

    return jsonify(sanitize({
        "jobs":  [j.model_dump() for j in jobs],
        "nodes": [n.model_dump() for n in nodes],
    }))


@app.route("/api/schedule", methods=["POST"])
def schedule():
    body           = request.get_json(force=True)
    jobs_data      = body.get("jobs", [])
    nodes_data     = body.get("nodes", [])
    scheduler_name = body.get("scheduler", "QIPS")
    weights        = body.get("weights", [1.0, 1.2, 2.5, 1.0, 1.2, 2.0])
    n_particles    = int(body.get("n_particles", 30))
    max_iterations = int(body.get("max_iterations", 100))
    beta           = float(body.get("beta", 0.75))

    jobs  = [Job(**j) for j in jobs_data]
    nodes = [Node(
        id=n["id"],
        total_cpu=n["total_cpu"],
        total_mem=n["total_mem"],
        available_cpu=n.get("available_cpu", n["total_cpu"]),
        available_mem=n.get("available_mem", n["total_mem"]),
        data_blocks=n.get("data_blocks", []),
    ) for n in nodes_data]

    history = []

    if scheduler_name == "QIPS":
        sched = QPSOScheduler(n_particles=n_particles, max_iter=max_iterations, beta=beta)
        order, fit, history = sched.optimize(jobs, nodes, weights)

    elif scheduler_name in SCHEDULER_MAP and SCHEDULER_MAP[scheduler_name] is not None:
        fn    = SCHEDULER_MAP[scheduler_name]
        order = fn(jobs, nodes, weights, {})
        fit   = 0.0

    else:
        return jsonify({"error": f"Unknown scheduler: {scheduler_name}"}), 400

    metrics     = compute_all_metrics(order, jobs, nodes)
    job_results = metrics.pop("job_results")

    return jsonify(sanitize({
        "scheduler":       scheduler_name,
        "order_indices":   [int(i) for i in order],
        "ordered_ids":     [jobs[i].id for i in order],
        "metrics":         metrics,
        "job_results":     job_results,
        "fitness_history": [round(float(v), 4) for v in history],
        "final_fitness":   round(float(fit), 4),
    }))


@app.route("/api/compare", methods=["POST"])
def compare():
    body       = request.get_json(force=True)
    jobs_data  = body.get("jobs", [])
    nodes_data = body.get("nodes", [])
    weights    = body.get("weights", [1.0, 1.2, 2.5, 1.0, 1.2, 2.0])
    n_particles    = int(body.get("n_particles", 30))
    max_iterations = int(body.get("max_iterations", 100))

    jobs  = [Job(**j) for j in jobs_data]
    nodes = [Node(
        id=n["id"],
        total_cpu=n["total_cpu"],
        total_mem=n["total_mem"],
        available_cpu=n.get("available_cpu", n["total_cpu"]),
        available_mem=n.get("available_mem", n["total_mem"]),
        data_blocks=n.get("data_blocks", []),
    ) for n in nodes_data]

    results = run_all_eight_schedulers(
        jobs, nodes, weights,
        qpso_params={"n_particles": n_particles, "max_iter": max_iterations, "seed": 42},
    )

    # Remove job_results from metrics (too large for compare payload)
    for name in results:
        results[name]["metrics"].pop("job_results", None)

    # Return in canonical paper order
    ordered = {s: results[s] for s in SCHEDULER_ORDER if s in results}
    return jsonify(sanitize(ordered))


# ─────────────────────────────────────────────────────────────
# Entry point (local dev only — Vercel uses the `app` object)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)