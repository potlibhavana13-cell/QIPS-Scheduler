"""
tests/test_api.py
Integration tests for the FastAPI QPSO service endpoints.

Run:
  pytest tests/test_api.py -v
"""
import pytest
from httpx import AsyncClient, ASGITransport
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qpso_engine.service import app

SAMPLE_JOBS = [
    {"id":"J-01","name":"WordCount","priority":0.9,"estimated_runtime":30,
     "cpu_demand":2,"mem_demand":1024,"deadline_slack":60,"input_size_gb":1.0},
    {"id":"J-02","name":"PageRank","priority":0.3,"estimated_runtime":120,
     "cpu_demand":4,"mem_demand":2048,"deadline_slack":300,"input_size_gb":5.0},
    {"id":"J-03","name":"TeraSort","priority":0.7,"estimated_runtime":60,
     "cpu_demand":2,"mem_demand":1024,"deadline_slack":90,"input_size_gb":2.0},
]
SAMPLE_NODES = [
    {"id":"node-1","total_cpu":8,"total_mem":8192,"available_cpu":8,
     "available_mem":8192,"data_blocks":["J-01","J-03"]},
    {"id":"node-2","total_cpu":8,"total_mem":8192,"available_cpu":8,
     "available_mem":8192,"data_blocks":["J-02"]},
    {"id":"node-3","total_cpu":8,"total_mem":8192,"available_cpu":8,
     "available_mem":8192,"data_blocks":[]},
]


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_optimize_returns_valid_order():
    payload = {
        "jobs": SAMPLE_JOBS, "nodes": SAMPLE_NODES,
        "weights": [1,1,1.5,1,1.2,1.3],
        "n_particles": 10, "max_iterations": 20, "beta": 0.75,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/optimize", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert sorted(data["ordered_job_ids"]) == ["J-01","J-02","J-03"]
    assert "metrics" in data
    assert len(data["fitness_history"]) == 21  # 20 iterations + initial


@pytest.mark.asyncio
async def test_optimize_metrics_keys():
    payload = {
        "jobs": SAMPLE_JOBS, "nodes": SAMPLE_NODES,
        "weights": [1,1,1,1,1,1],
        "n_particles": 5, "max_iterations": 10, "beta": 0.75,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/optimize", json=payload)
    m = r.json()["metrics"]
    for key in ["makespan","avg_latency","deadline_penalty",
                "resource_usage_pct","data_locality_pct","priority_satisfaction"]:
        assert key in m, f"Missing metric: {key}"


@pytest.mark.asyncio
async def test_compare_returns_all_schedulers():
    payload = {"jobs": SAMPLE_JOBS, "nodes": SAMPLE_NODES,
               "weights": [1,1,1,1,1,1],
               "schedulers": ["QIPS","FIFO","Fair","Capacity"]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/compare", json=payload)
    assert r.status_code == 200
    data = r.json()["results"]
    assert set(data.keys()) == {"QIPS","FIFO","Fair","Capacity"}


@pytest.mark.asyncio
async def test_optimize_empty_jobs_returns_400():
    payload = {"jobs": [], "nodes": SAMPLE_NODES,
               "weights": [1,1,1,1,1,1], "n_particles": 5, "max_iterations": 10}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/optimize", json=payload)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_sample_workload_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/sample-workload")
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data and "nodes" in data
