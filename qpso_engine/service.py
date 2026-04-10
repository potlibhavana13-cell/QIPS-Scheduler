"""
qpso_engine/service.py
FastAPI REST service exposing the QPSO optimizer.

Endpoints:
  POST /optimize          — run QIPS on a job set, return optimized order
  POST /compare           — run QIPS + all baselines, return side-by-side metrics
  GET  /health            — health check
  GET  /docs              — auto-generated OpenAPI docs (built-in FastAPI)

Run:
  uvicorn qpso_engine.service:app --reload --port 8080
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from .models import ScheduleRequest, ScheduleResponse, CompareRequest, CompareResponse, JobResult
from .qpso import QPSOScheduler, run_all_schedulers
from .fitness import compute_all_metrics

app = FastAPI(
    title="QIPS — Quantum-Inspired Priority Scheduler",
    description="REST API for the QPSO-based Hadoop YARN scheduler (paper implementation).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "QIPS QPSO Engine"}


@app.post("/optimize", response_model=ScheduleResponse)
def optimize(req: ScheduleRequest):
    """
    Run the QPSO optimizer on the provided jobs and nodes.
    Returns the optimized job execution order and all scheduling metrics.
    """
    if not req.jobs:
        raise HTTPException(status_code=400, detail="At least one job required.")
    if not req.nodes:
        raise HTTPException(status_code=400, detail="At least one node required.")

    t0 = time.perf_counter()

    scheduler = QPSOScheduler(
        n_particles=req.n_particles,
        max_iter=req.max_iterations,
        beta=req.beta,
    )
    best_order, best_fitness, history = scheduler.optimize(
        req.jobs, req.nodes, req.weights
    )

    elapsed = time.perf_counter() - t0
    metrics = compute_all_metrics(best_order, req.jobs, req.nodes)

    job_results = [
        JobResult(**jr) for jr in metrics.pop("job_results")
    ]

    metrics["optimization_time_ms"] = round(elapsed * 1000, 1)

    return ScheduleResponse(
        ordered_job_ids=[req.jobs[i].id for i in best_order],
        job_results=job_results,
        metrics=metrics,
        fitness_history=[round(v, 4) for v in history],
        final_fitness=round(best_fitness, 4),
        iterations_run=len(history) - 1,
        scheduler="QIPS",
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """
    Run QIPS + baseline schedulers (FIFO, Fair, Capacity) on the same workload.
    Returns per-scheduler metrics for direct comparison.
    """
    if not req.jobs:
        raise HTTPException(status_code=400, detail="At least one job required.")

    results = run_all_schedulers(req.jobs, req.nodes, req.weights)

    # Filter to requested schedulers only
    filtered = {k: v for k, v in results.items() if k in req.schedulers}
    return CompareResponse(results=filtered)


@app.get("/sample-workload")
def sample_workload():
    """Returns a sample request body you can paste into /optimize for testing."""
    return {
        "jobs": [
            {"id": "J-01", "name": "WordCount",   "priority": 0.9, "estimated_runtime": 45,
             "cpu_demand": 2, "mem_demand": 1024, "deadline_slack": 80,  "input_size_gb": 2.0, "local_node_id": "node-1"},
            {"id": "J-02", "name": "PageRank",    "priority": 0.3, "estimated_runtime": 120,
             "cpu_demand": 4, "mem_demand": 2048, "deadline_slack": 300, "input_size_gb": 5.0, "local_node_id": "node-2"},
            {"id": "J-03", "name": "TeraSort",    "priority": 0.7, "estimated_runtime": 60,
             "cpu_demand": 2, "mem_demand": 1024, "deadline_slack": 90,  "input_size_gb": 1.5, "local_node_id": "node-1"},
            {"id": "J-04", "name": "HiveQuery",   "priority": 0.5, "estimated_runtime": 30,
             "cpu_demand": 1, "mem_demand": 512,  "deadline_slack": 200, "input_size_gb": 0.8, "local_node_id": "node-3"},
        ],
        "nodes": [
            {"id": "node-1", "total_cpu": 8, "total_mem": 8192, "available_cpu": 8,
             "available_mem": 8192, "data_blocks": ["J-01", "J-03"]},
            {"id": "node-2", "total_cpu": 8, "total_mem": 8192, "available_cpu": 8,
             "available_mem": 8192, "data_blocks": ["J-02"]},
            {"id": "node-3", "total_cpu": 8, "total_mem": 8192, "available_cpu": 8,
             "available_mem": 8192, "data_blocks": ["J-04"]},
        ],
        "weights": [1.0, 1.0, 1.5, 1.0, 1.2, 1.3],
        "n_particles": 30,
        "max_iterations": 100,
        "beta": 0.75,
    }
