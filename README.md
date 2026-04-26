# QIPS — Quantum-Inspired Priority Scheduler for Hadoop YARN

> Implementation and simulation of the paper:  
> **"Optimizing Priority Scheduling in Hadoop for Resource Utilization using Quantum Particle Swarm Optimization Technique"**  
> Bhavana P, Shashikumar D R — *International Journal of Advances in Intelligent Informatics*, 2018

---

## What this repo contains

| Folder | What it does |
|---|---|
| `qpso_engine/` | Pure Python QPSO optimizer + REST API (FastAPI) |
| `simulator/` | Interactive browser-based demo (Flask + HTML/JS) |
| `yarn_plugin/` | Java skeleton for the real YARN ResourceManager plugin |
| `notebooks/` | Jupyter notebooks for analysis and benchmarking |
| `tests/` | Unit + integration tests |
| `scripts/` | Cluster setup and benchmark automation |
| `demo/` | Standalone HTML demo (no server needed) |

---

## Quick start (simulation — no Hadoop needed)

```bash
# 1. Clone and enter
git clone https://github.com/yourname/qips-scheduler.git
cd qips-scheduler

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the interactive simulator
python simulator/app.py
# Open http://localhost:5000

# 5. Or run the QPSO REST API standalone
uvicorn qpso_engine.service:app --reload --port 8080
# API docs at http://localhost:8080/docs
```

### Standalone demo (zero dependencies)
Open `demo/index.html` directly in any browser — no server, no install.

---

## Architecture

```
Browser / Client
      │
      ▼
simulator/app.py  (Flask — serves the UI + proxies to QPSO)
      │
      ▼  REST  POST /optimize
qpso_engine/service.py  (FastAPI — QPSO optimizer)
      │
      ├── qpso_engine/qpso.py        (core QPSO algorithm, Eq. 2–3)
      ├── qpso_engine/fitness.py     (multi-objective fitness, Eq. 1)
      └── qpso_engine/models.py      (Pydantic data models)

yarn_plugin/  (Java — plugs into real Hadoop 3.3.4 YARN)
      ├── QIPSScheduler.java         (extends AbstractYarnScheduler)
      ├── JobClassifier.java
      ├── TaskProfiler.java
      └── QpsoRestClient.java        (calls the Python service)
```

---

## Fitness function (Equation 1)

```
F(Pᵢ) = w₁·T_makespan + w₂·T_latency + w₃·P_deadline
         − w₄·R_usage  − w₅·L_data    − w₆·S_priority
```

| Term | Meaning | Direction |
|---|---|---|
| T_makespan | Total job completion time | Minimize |
| T_latency | Average waiting time | Minimize |
| P_deadline | Penalty for missed deadlines (ReLU) | Minimize |
| R_usage | CPU/memory utilization ratio | Maximize |
| L_data | Fraction of data-local tasks | Maximize |
| S_priority | Priority satisfaction score | Maximize |

## QPSO position update (Equation 2)

```
x_ij(t+1) = p_ij(t) ± β · |m_j(t) − x_ij(t)| · ln(1/u)
u ~ U(0,1)
```

---

## Running benchmarks

```bash
# Compare all schedulers on synthetic workloads
python scripts/benchmark.py --jobs 20 --runs 10

# Run Jupyter analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Reproducing paper results

See `notebooks/reproduce_paper_results.ipynb` for a step-by-step walkthrough that generates the same comparison table (Table 1) from the paper.

---

## Requirements

- Python 3.10+
- Java 11+ (for YARN plugin only)
- Maven 3.8+ (for YARN plugin only)
- Hadoop 3.3.4 (for real cluster deployment)

---

## Live Demo

🔗 https://qips-demo.vercel.app

 **Note:** This is a serverless deployment on Vercel, so it may have certain limitations. At times, updated values might not reflect immediately.

### 🔄If values are not updating:
- Perform a **hard refresh** in your browser:
  - **Windows/Linux:** `Ctrl + Shift + R` or `Ctrl + F5`
  - **Mac:** `Cmd + Shift + R`
- Then continue testing the application.

---

## Important
This is a **demo simulation**.
For **more accurate and consistent results**, it is recommended to run the project locally.

## Citation

```bibtex
@article{bhavana2018qips,
  title   = {Optimizing Priority Scheduling in Hadoop for Resource Utilization
             using Quantum Particle Swarm Optimization Technique},
  author  = {Bhavana P and Shashikumar D R},
  journal = {International Journal of Advances in Intelligent Informatics},
  volume  = {4},
  number  = {2},
  year    = {2018}
}
```
