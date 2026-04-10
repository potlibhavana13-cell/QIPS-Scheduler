"""
qpso_engine/models.py
Pure-Python dataclasses (no Pydantic required).
Pydantic is used only when the FastAPI service is running (optional extra).
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Job:
    id: str
    name: str
    priority: float          # normalized 0–1
    estimated_runtime: float # seconds
    cpu_demand: int
    mem_demand: int          # MB
    deadline_slack: float    # seconds until deadline
    input_size_gb: float
    local_node_id: Optional[str] = None

    def model_dump(self):
        return self.__dict__.copy()


@dataclass
class Node:
    id: str
    total_cpu: int
    total_mem: int           # MB
    available_cpu: int
    available_mem: int
    data_blocks: List[str] = field(default_factory=list)

    def model_dump(self):
        return self.__dict__.copy()


def job_from_dict(d: dict) -> Job:
    return Job(
        id=d["id"], name=d["name"],
        priority=float(d["priority"]),
        estimated_runtime=float(d["estimated_runtime"]),
        cpu_demand=int(d["cpu_demand"]),
        mem_demand=int(d["mem_demand"]),
        deadline_slack=float(d["deadline_slack"]),
        input_size_gb=float(d["input_size_gb"]),
        local_node_id=d.get("local_node_id"),
    )


def node_from_dict(d: dict) -> Node:
    return Node(
        id=d["id"],
        total_cpu=int(d["total_cpu"]),
        total_mem=int(d["total_mem"]),
        available_cpu=int(d["available_cpu"]),
        available_mem=int(d["available_mem"]),
        data_blocks=list(d.get("data_blocks", [])),
    )
