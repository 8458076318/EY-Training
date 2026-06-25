"""Observability for a multi-agent system.

This script upgrades the starter "print step X/Y" example into a small,
offline-friendly telemetry demo. It emits structured JSON events with a
run-level trace_id, per-agent span_id, throttled progress updates, failure
localization, and a final run summary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def new_id() -> str:
    return uuid.uuid4().hex


def pct(value: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return round((value / total) * 100.0, 2)


def safe_rate(steps: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return float(steps)
    return round(steps / elapsed_seconds, 3)


class EventSink:
    def __init__(self, trace_id: str, run_span_id: str, output_path: Optional[Path] = None):
        self.trace_id = trace_id
        self.run_span_id = run_span_id
        self.output_path = output_path
        self._handle = output_path.open("a", encoding="utf-8") if output_path else None

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def emit(self, event: str, span_id: str, agent: Optional[str] = None, **fields) -> Dict[str, object]:
        record: Dict[str, object] = {
            "timestamp": utc_timestamp(),
            "trace_id": self.trace_id,
            "span_id": span_id,
            "event": event,
        }
        if agent is not None:
            record["agent"] = agent
        record.update(fields)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        print(line)
        if self._handle:
            self._handle.write(line + "\n")
            self._handle.flush()
        return record


@dataclass
class Agent:
    name: str
    steps: int
    fail_at_step: Optional[int] = None

    def run(
        self,
        sink: EventSink,
        parent_span_id: str,
        completed_steps_before_agent: int,
        total_pipeline_steps: int,
        pipeline_started_at: float,
    ) -> int:
        span_id = new_id()
        agent_started_at = time.perf_counter()
        agent_checkpoint = max(1, math.ceil(self.steps * 0.25))
        next_progress_step = agent_checkpoint
        sink.emit(
            "agent_started",
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent=self.name,
            total_steps=self.steps,
            completed_steps_before_agent=completed_steps_before_agent,
            pipeline_percent_complete=pct(completed_steps_before_agent, total_pipeline_steps),
            throughput_steps_per_sec=safe_rate(completed_steps_before_agent, time.perf_counter() - pipeline_started_at),
        )

        completed_in_agent = 0
        for step in range(1, self.steps + 1):
            time.sleep(random.uniform(0.05, 0.2))

            if self.fail_at_step is not None and step == self.fail_at_step:
                elapsed_pipeline = time.perf_counter() - pipeline_started_at
                sink.emit(
                    "agent_failed",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    agent=self.name,
                    step=step,
                    total_steps=self.steps,
                    completed_steps=completed_steps_before_agent + completed_in_agent,
                    error=f"{self.name} failed at step {step}",
                    pipeline_percent_complete=pct(
                        completed_steps_before_agent + completed_in_agent, total_pipeline_steps
                    ),
                    throughput_steps_per_sec=safe_rate(
                        completed_steps_before_agent + completed_in_agent, elapsed_pipeline
                    ),
                    duration_seconds=round(time.perf_counter() - agent_started_at, 3),
                )
                raise RuntimeError(f"{self.name} failed at step {step}")

            completed_in_agent += 1
            if completed_in_agent >= next_progress_step or completed_in_agent == self.steps:
                elapsed_pipeline = time.perf_counter() - pipeline_started_at
                sink.emit(
                    "agent_progress",
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    agent=self.name,
                    step=completed_in_agent,
                    total_steps=self.steps,
                    completed_steps=completed_steps_before_agent + completed_in_agent,
                    agent_percent_complete=pct(completed_in_agent, self.steps),
                    pipeline_percent_complete=pct(
                        completed_steps_before_agent + completed_in_agent, total_pipeline_steps
                    ),
                    throughput_steps_per_sec=safe_rate(
                        completed_steps_before_agent + completed_in_agent, elapsed_pipeline
                    ),
                    eta_seconds=round(
                        max(0.0, (total_pipeline_steps - (completed_steps_before_agent + completed_in_agent)))
                        / max(0.001, safe_rate(completed_steps_before_agent + completed_in_agent, elapsed_pipeline)),
                        3,
                    ),
                )
                next_progress_step += agent_checkpoint

        elapsed_pipeline = time.perf_counter() - pipeline_started_at
        sink.emit(
            "agent_completed",
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent=self.name,
            total_steps=self.steps,
            completed_steps=completed_steps_before_agent + completed_in_agent,
            duration_seconds=round(time.perf_counter() - agent_started_at, 3),
            pipeline_percent_complete=pct(
                completed_steps_before_agent + completed_in_agent, total_pipeline_steps
            ),
            throughput_steps_per_sec=safe_rate(
                completed_steps_before_agent + completed_in_agent, elapsed_pipeline
            ),
        )
        return completed_in_agent


class Orchestrator:
    def __init__(self, agents: Iterable[Agent], sink: EventSink):
        self.agents = list(agents)
        self.sink = sink

    def run(self) -> Dict[str, object]:
        run_started_at = time.perf_counter()
        total_steps = sum(agent.steps for agent in self.agents)
        run_status = "completed"
        completed_agents = 0
        completed_steps = 0
        failed_agent = None
        failed_step = None

        self.sink.emit(
            "run_started",
            span_id=self.sink.run_span_id,
            agent="orchestrator",
            total_agents=len(self.agents),
            total_steps=total_steps,
        )

        for agent in self.agents:
            try:
                completed = agent.run(
                    sink=self.sink,
                    parent_span_id=self.sink.run_span_id,
                    completed_steps_before_agent=completed_steps,
                    total_pipeline_steps=total_steps,
                    pipeline_started_at=run_started_at,
                )
                completed_steps += completed
                completed_agents += 1
            except RuntimeError as exc:
                run_status = "failed"
                failed_agent = agent.name
                failed_step = agent.fail_at_step
                completed_steps = completed_steps + max(0, (agent.fail_at_step or 1) - 1)
                elapsed_pipeline = time.perf_counter() - run_started_at
                self.sink.emit(
                    "run_summary",
                    span_id=self.sink.run_span_id,
                    agent="orchestrator",
                    status=run_status,
                    total_agents=len(self.agents),
                    agents_completed=completed_agents,
                    total_steps=total_steps,
                    steps_completed=completed_steps,
                    failed_agent=failed_agent,
                    failed_step=failed_step,
                    duration_seconds=round(elapsed_pipeline, 3),
                    pipeline_percent_complete=pct(completed_steps, total_steps),
                    throughput_steps_per_sec=safe_rate(completed_steps, elapsed_pipeline),
                    error=str(exc),
                )
                return {
                    "status": run_status,
                    "total_agents": len(self.agents),
                    "agents_completed": completed_agents,
                    "total_steps": total_steps,
                    "steps_completed": completed_steps,
                    "failed_agent": failed_agent,
                    "failed_step": failed_step,
                    "duration_seconds": round(elapsed_pipeline, 3),
                }

        elapsed_pipeline = time.perf_counter() - run_started_at
        self.sink.emit(
            "run_summary",
            span_id=self.sink.run_span_id,
            agent="orchestrator",
            status=run_status,
            total_agents=len(self.agents),
            agents_completed=completed_agents,
            total_steps=total_steps,
            steps_completed=completed_steps,
            failed_agent=None,
            failed_step=None,
            duration_seconds=round(elapsed_pipeline, 3),
            pipeline_percent_complete=pct(completed_steps, total_steps),
            throughput_steps_per_sec=safe_rate(completed_steps, elapsed_pipeline),
        )
        return {
            "status": run_status,
            "total_agents": len(self.agents),
            "agents_completed": completed_agents,
            "total_steps": total_steps,
            "steps_completed": completed_steps,
            "failed_agent": None,
            "failed_step": None,
            "duration_seconds": round(elapsed_pipeline, 3),
        }


def build_agents(fail_agent: Optional[str], fail_step: Optional[int]) -> List[Agent]:
    agents = [
        Agent("Planner", 3),
        Agent("Researcher", 6),
        Agent("Writer", 4),
        Agent("Reviewer", 2),
    ]
    if fail_agent and fail_step:
        for agent in agents:
            if agent.name.lower() == fail_agent.lower():
                agent.fail_at_step = fail_step
                break
    return agents


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent observability demo")
    parser.add_argument("--fail-agent", help="Agent name to fail for the demo", default=None)
    parser.add_argument("--fail-step", help="Step number at which the selected agent fails", type=int, default=None)
    parser.add_argument(
        "--trace-file",
        help="Optional JSONL file to append events to in addition to stdout",
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    trace_id = new_id()
    run_span_id = new_id()
    sink = EventSink(trace_id=trace_id, run_span_id=run_span_id, output_path=Path(args.trace_file) if args.trace_file else None)
    try:
        agents = build_agents(args.fail_agent, args.fail_step)
        orchestrator = Orchestrator(agents, sink)
        summary = orchestrator.run()
        print(json.dumps(summary, sort_keys=True))
        return 0
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())


# Submission note:
# Capture agent token usage/cost per span next, because it helps explain slow or expensive hops more than step counts alone.
# In a client environment, I would ship these JSON events to stdout for local debugging and to an OpenTelemetry or log pipeline for aggregation.
