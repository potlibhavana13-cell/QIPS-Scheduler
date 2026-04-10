"""
scripts/benchmark.py
Compare QIPS vs baseline schedulers on randomized synthetic workloads.

Usage:
  python scripts/benchmark.py --jobs 10 --runs 20 --output results.csv
"""
import sys, os, argparse, random, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qpso_engine import QPSOScheduler, Job, Node, compute_all_metrics
from qpso_engine.qpso import fifo_schedule, fair_schedule, capacity_schedule

SCHEDULER_FNS = {
    "FIFO":     lambda jobs, nodes, w: fifo_schedule(jobs),
    "Fair":     lambda jobs, nodes, w: fair_schedule(jobs),
    "Capacity": lambda jobs, nodes, w: capacity_schedule(jobs),
    "QIPS":     None,  # handled separately
}

JOB_NAMES = ["WordCount","PageRank","TeraSort","HiveQuery","SparkMLlib",
              "HBase Scan","Pig Latin","MapReduce ETL","Flume","Sqoop"]


def make_workload(n_jobs, n_nodes, seed):
    rng = random.Random(seed)
    nodes = [
        Node(id=f"node-{i+1}", total_cpu=8, total_mem=8192,
             available_cpu=8, available_mem=8192, data_blocks=[])
        for i in range(n_nodes)
    ]
    jobs = []
    for i in range(n_jobs):
        local_node = rng.randint(0, n_nodes - 1)
        runtime = rng.randint(20, 180)
        j = Job(
            id=f"J-{i+1:02d}",
            name=JOB_NAMES[i % len(JOB_NAMES)],
            priority=rng.choice([1.0, 0.75, 0.4, 0.15]),
            estimated_runtime=float(runtime),
            cpu_demand=rng.randint(1, 6),
            mem_demand=rng.choice([512, 1024, 2048, 4096]),
            deadline_slack=float(rng.randint(int(runtime * 1.1), int(runtime * 4))),
            input_size_gb=round(rng.uniform(0.5, 8.0), 1),
            local_node_id=nodes[local_node].id,
        )
        jobs.append(j)
        nodes[local_node].data_blocks.append(j.id)
    return jobs, nodes


def run_benchmark(n_jobs=10, n_runs=10, weights=None, qpso_params=None):
    if weights is None:
        weights = [1.0, 1.0, 1.5, 1.0, 1.2, 1.3]
    if qpso_params is None:
        qpso_params = {"n_particles": 30, "max_iter": 100, "beta": 0.75}

    aggregated = {name: [] for name in [*SCHEDULER_FNS.keys()]}

    print(f"\nBenchmark: {n_runs} runs × {n_jobs} jobs × 3 nodes")
    print("-" * 60)

    for run in range(n_runs):
        seed = run * 42
        jobs, nodes = make_workload(n_jobs, 3, seed)

        for name in SCHEDULER_FNS:
            if name == "QIPS":
                sched = QPSOScheduler(**qpso_params)
                order, _, _ = sched.optimize(jobs, nodes, weights)
            else:
                order = SCHEDULER_FNS[name](jobs, nodes, weights)

            m = compute_all_metrics(order, jobs, nodes)
            m.pop("job_results", None)
            m["scheduler"] = name
            m["run"] = run
            aggregated[name].append(m)

        print(f"  Run {run+1}/{n_runs} done", end="\r")

    print("\n")
    return aggregated


def print_summary(results):
    metrics_to_show = ["makespan", "avg_latency", "deadline_penalty",
                       "resource_usage_pct", "data_locality_pct", "priority_satisfaction"]
    header = f"{'Scheduler':<12}" + "".join(f"{m:>18}" for m in metrics_to_show)
    print(header)
    print("-" * len(header))

    import statistics
    for name, runs in results.items():
        row = f"{name:<12}"
        for metric in metrics_to_show:
            vals = [r[metric] for r in runs]
            mean = statistics.mean(vals)
            row += f"{mean:>18.2f}"
        print(row)


def save_csv(results, path):
    all_rows = []
    for name, runs in results.items():
        all_rows.extend(runs)
    if not all_rows:
        return
    keys = [k for k in all_rows[0] if k not in ("job_results",)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in keys})
    print(f"Results saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",   type=int, default=10)
    parser.add_argument("--runs",   type=int, default=10)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    results = run_benchmark(n_jobs=args.jobs, n_runs=args.runs)
    print_summary(results)
    save_csv(results, args.output)
