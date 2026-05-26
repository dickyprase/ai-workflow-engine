"""Task decomposition, scheduling, and execution management."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """Task execution priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class TaskResult:
    """Result of a completed task execution."""

    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    cost: float = 0.0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A unit of work within a workflow pipeline."""

    name: str
    handler: Callable[..., Awaitable[Any]] | None = None
    prompt: str | None = None
    model: str | None = None
    depends_on: list[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 30.0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: TaskResult | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Task):
            return self.id == other.id
        return NotImplemented


class TaskDecomposer:
    """Breaks complex objectives into atomic, parallelizable subtasks."""

    def __init__(self, max_depth: int = 3, max_subtasks: int = 10):
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks

    async def decompose(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        model_fn: Callable[..., Awaitable[str]] | None = None,
    ) -> list[Task]:
        """Decompose a high-level objective into executable subtasks.

        If a model_fn is provided, uses AI to intelligently break down the task.
        Otherwise, falls back to rule-based decomposition.
        """
        if model_fn:
            return await self._ai_decompose(objective, context, model_fn)
        return self._rule_based_decompose(objective, context)

    async def _ai_decompose(
        self,
        objective: str,
        context: dict[str, Any] | None,
        model_fn: Callable[..., Awaitable[str]],
    ) -> list[Task]:
        """Use an LLM to decompose the objective into subtasks."""
        decomposition_prompt = f"""Break down the following objective into atomic, executable subtasks.
Each subtask should be independent where possible, or have clear dependencies.

Objective: {objective}
Context: {context or 'None provided'}

Return a JSON array of tasks with fields: name, description, depends_on (list of task names), priority (0-4).
Return ONLY the JSON array, no other text."""

        try:
            response = await model_fn(decomposition_prompt)
            import json

            tasks_data = json.loads(response)
            tasks = []
            for t in tasks_data[: self.max_subtasks]:
                task = Task(
                    name=t["name"],
                    prompt=t.get("description", ""),
                    depends_on=t.get("depends_on", []),
                    priority=TaskPriority(t.get("priority", 2)),
                )
                tasks.append(task)
            return tasks
        except Exception as e:
            logger.warning(f"AI decomposition failed, falling back to rules: {e}")
            return self._rule_based_decompose(objective, context)

    def _rule_based_decompose(
        self, objective: str, context: dict[str, Any] | None
    ) -> list[Task]:
        """Simple rule-based decomposition for when AI is unavailable."""
        # Split by common delimiters and create sequential tasks
        parts = [p.strip() for p in objective.split(";") if p.strip()]
        if len(parts) == 1:
            parts = [p.strip() for p in objective.split(" then ") if p.strip()]

        tasks = []
        prev_name = None
        for i, part in enumerate(parts):
            name = f"step_{i + 1}"
            task = Task(
                name=name,
                prompt=part,
                depends_on=[prev_name] if prev_name else [],
                priority=TaskPriority.NORMAL,
            )
            tasks.append(task)
            prev_name = name

        return tasks if tasks else [Task(name="main", prompt=objective)]


class TaskScheduler:
    """DAG-based task scheduler with priority queue and concurrency control."""

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._running: set[str] = set()

    def add_task(self, task: Task) -> None:
        """Register a task with the scheduler."""
        self._tasks[task.name] = task

    def add_tasks(self, tasks: list[Task]) -> None:
        """Register multiple tasks."""
        for task in tasks:
            self.add_task(task)

    async def execute_all(
        self,
        executor: Callable[[Task, dict[str, TaskResult]], Awaitable[TaskResult]],
    ) -> dict[str, TaskResult]:
        """Execute all tasks respecting dependencies and concurrency limits.

        Uses topological ordering with parallel execution of independent tasks.
        """
        pending = set(self._tasks.keys())
        completed: set[str] = set()

        while pending:
            # Find tasks whose dependencies are all satisfied
            ready = [
                name
                for name in pending
                if all(dep in completed for dep in self._tasks[name].depends_on)
            ]

            if not ready:
                # Check for circular dependencies
                raise RuntimeError(
                    f"Circular dependency detected. Stuck tasks: {pending}"
                )

            # Sort by priority and execute in parallel
            ready.sort(key=lambda n: self._tasks[n].priority.value)

            # Execute ready tasks concurrently (up to semaphore limit)
            coros = [
                self._execute_with_semaphore(self._tasks[name], executor)
                for name in ready
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for name, result in zip(ready, results):
                if isinstance(result, Exception):
                    self._results[name] = TaskResult(
                        task_id=self._tasks[name].id,
                        status=TaskStatus.FAILED,
                        error=str(result),
                    )
                else:
                    self._results[name] = result

                pending.discard(name)
                completed.add(name)

        return self._results

    async def _execute_with_semaphore(
        self,
        task: Task,
        executor: Callable[[Task, dict[str, TaskResult]], Awaitable[TaskResult]],
    ) -> TaskResult:
        """Execute a task with concurrency limiting."""
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            self._running.add(task.name)
            logger.info(f"Executing task: {task.name} (priority={task.priority.name})")

            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    executor(task, self._results),
                    timeout=task.timeout,
                )
                task.status = TaskStatus.COMPLETED
                result.duration_ms = (time.perf_counter() - start) * 1000
                return result
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=f"Task timed out after {task.timeout}s",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                task.status = TaskStatus.FAILED
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            finally:
                self._running.discard(task.name)

    @property
    def progress(self) -> dict[str, int]:
        """Return execution progress summary."""
        statuses = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for task in self._tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
                statuses["pending"] += 1
            elif task.status == TaskStatus.RUNNING:
                statuses["running"] += 1
            elif task.status == TaskStatus.COMPLETED:
                statuses["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                statuses["failed"] += 1
        return statuses
