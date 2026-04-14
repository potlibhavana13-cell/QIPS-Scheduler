"""
simulator/app.py
Flask web app — serves the interactive QIPS simulator UI.
Modified for deployment (e.g., Vercel / serverless environments)
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

from qpso_engine import QPSOScheduler, run_all_schedulers, Job, Node, compute_all_metrics
from qpso_engine.qpso import fifo_schedule, fair_schedule, capacity_schedule

# ✅ Fix template path for deployment
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────
def sanitize(obj):
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


JOB_NAMES = [
    "WordCount", "PageRank", "TeraSort", "HiveQuery", "SparkMLlib",
    "HBase Scan", "Pig Latin", "MapReduce ETL", "Flume Ingest", "Sqoop Import",
]

PRIORITY_TIERS = [
    ("Critical", 1.0), ("High", 0.75), ("Medium", 0.4), ("Low", 0.15)
]


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
            data_blocks=[]
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
        int(body.get("n_jobs", 6)),
        int(body.get("n_nodes", 3)),
        body.get("seed")
    )

    return jsonify(sanitize({
        "jobs": [j.model_dump() for j in jobs],
        "nodes": [n.model_dump() for n in nodes],
    }))


@app.route("/api/schedule", methods=["POST"])
def schedule():
    body = request.get_json(force=True)

    jobs = [Job(**j) for j in body.get("jobs", [])]
    nodes = [Node(**n) for n in body.get("nodes", [])]

    scheduler_name = body.get("scheduler", "QIPS")
    weights = body.get("weights", [1.0, 1.2, 2.5, 1.0, 1.2, 2.0])

    if scheduler_name == "QIPS":
        sched = QPSOScheduler(
            n_particles=int(body.get("n_particles", 30)),
            max_iter=int(body.get("max_iterations", 100)),
            beta=float(body.get("beta", 0.75))
        )
        order, fit, history = sched.optimize(jobs, nodes, weights)

    elif scheduler_name == "FIFO":
        order, fit, history = fifo_schedule(jobs), 0.0, []

    elif scheduler_name == "Fair":
        order, fit, history = fair_schedule(jobs), 0.0, []

    elif scheduler_name == "Capacity":
        order, fit, history = capacity_schedule(jobs), 0.0, []

    else:
        return jsonify({"error": f"Unknown scheduler: {scheduler_name}"}), 400

    metrics = compute_all_metrics(order, jobs, nodes)
    job_results = metrics.pop("job_results")

    return jsonify(sanitize({
        "scheduler": scheduler_name,

        # 🔥 ADD THIS (CRITICAL)
        "order_indices": order,

        # keep existing
        "ordered_ids": [jobs[i].id for i in order],

        "metrics": metrics,
        "job_results": job_results,
        "fitness_history": [round(float(v), 4) for v in history],
        "final_fitness": round(float(fit), 4),
    }))


@app.route("/api/compare", methods=["POST"])
def compare():
    body = request.get_json(force=True)

    jobs = [Job(**j) for j in body.get("jobs", [])]
    nodes = [Node(**n) for n in body.get("nodes", [])]

    results = run_all_schedulers(
        jobs,
        nodes,
        body.get("weights", [1.0, 1.2, 2.5, 1.0, 1.2, 2.0])
    )

    for name in results:
        results[name]["metrics"].pop("job_results", None)

    return jsonify(sanitize(results))

