"""
simulator/app.py
Flask web app — serves the interactive QIPS simulator UI.
Run:  python simulator/app.py  →  http://localhost:5000
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, jsonify, request
import random
import numpy as np

from qpso_engine import QPSOScheduler, run_all_schedulers, Job, Node, compute_all_metrics
from qpso_engine.qpso import fifo_schedule, fair_schedule, capacity_schedule

app = Flask(__name__)


def sanitize(obj):
    if isinstance(obj, dict):   return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return [sanitize(v) for v in obj.tolist()]
    return obj


JOB_NAMES = [
    "WordCount", "PageRank", "TeraSort", "HiveQuery", "SparkMLlib",
    "HBase Scan", "Pig Latin", "MapReduce ETL", "Flume Ingest", "Sqoop Import",
]
# Priority tiers ordered Critical → High → Medium → Low
PRIORITY_TIERS = [
    ("Critical", 1.0), ("High", 0.75), ("Medium", 0.4), ("Low", 0.15)
]


def generate_workload(n_jobs: int = 6, n_nodes: int = 3, seed: int | None = None):
    """
    Generate a synthetic workload. Each job gets a local_node_id and that
    node's data_blocks list is populated, enabling accurate locality scoring.

    Deadline distribution: ~1/3 tight (factor 1.2–1.5), ~1/3 medium (2–2.5),
    ~1/3 loose (3–4). This ensures some jobs are genuinely urgent, giving
    QIPS a real signal to exploit over FIFO/Fair/Capacity.
    """
    if seed is not None:
        random.seed(seed)

    nodes = [
        Node(
            id=f"Node-{i+1}",
            total_cpu=8, total_mem=8192,
            available_cpu=8, available_mem=8192,
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
        runtime  = random.randint(20, 180)
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/workload", methods=["POST"])
def new_workload():
    body    = request.get_json(force=True)
    n_jobs  = int(body.get("n_jobs", 6))
    n_nodes = int(body.get("n_nodes", 3))
    seed    = body.get("seed")
    jobs, nodes = generate_workload(n_jobs, n_nodes, seed)
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

    jobs  = [Job(**j)  for j in jobs_data]
    nodes = [Node(**n) for n in nodes_data]

    if scheduler_name == "QIPS":
        sched = QPSOScheduler(n_particles=n_particles, max_iter=max_iterations, beta=beta)
        order, fit, history = sched.optimize(jobs, nodes, weights)
    elif scheduler_name == "FIFO":
        order, fit, history = fifo_schedule(jobs), 0.0, []
    elif scheduler_name == "Fair":
        order, fit, history = fair_schedule(jobs), 0.0, []
    elif scheduler_name == "Capacity":
        order, fit, history = capacity_schedule(jobs), 0.0, []
    else:
        return jsonify({"error": f"Unknown scheduler: {scheduler_name}"}), 400

    metrics     = compute_all_metrics(order, jobs, nodes)
    job_results = metrics.pop("job_results")

    return jsonify(sanitize({
        "scheduler":       scheduler_name,
        "ordered_ids":     [jobs[i].id for i in order],
        "order_indices":   [int(i) for i in order],
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

    jobs  = [Job(**j)  for j in jobs_data]
    nodes = [Node(**n) for n in nodes_data]

    results = run_all_schedulers(jobs, nodes, weights)
    for name in results:
        results[name]["metrics"].pop("job_results", None)

    return jsonify(sanitize(results))


if __name__ == "__main__":
    app.run(debug=True, port=5000)