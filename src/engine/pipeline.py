"""Pipeline orchestration with DAG-based workflow execution."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from engine.tasks import Task, TaskResult, TaskStatus, TaskScheduler, TaskPriority
from engine.models import ModelRouter, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a pipeline instance."""

    name: str
    description: str = ""
    max_concurrency: int = 5
    timeout: float = 300.0
    retry_failed: bool = True
    collect_metrics: bool = True


@dataclass
class PipelineMetrics:
    """Execution metrics for a pipeline run."""

    total_duration_ms: float = 0.0
    total_cost: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tokens_used: int = 0
    models_used: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete result of a pipeline execution."""

    pipeline_id: str
    pipeline_name: str
    output: Any = None
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    success: bool = True
    error: str | None = None


@dataclass
class PipelineNode:
    """A node in the pipeline DAG."""

    task: Task
    condition: Callable[[dict[str, TaskResult]], bool] | None = None
    transform_input: Callable[[dict[str, TaskResult]], dict[str, Any]] | None = None
    on_error: Callable[[Exception], None] | None = None


class Pipeline:
    """DAG-based pipeline with conditional branching and error handling."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.id = str(uuid.uuid4())
        self._nodes: dict[str, PipelineNode] = {}
        self._error_handler: Callable[[Exception], None] | None = None

    @property
    def name(self) -> str:
        return self.config.name

    def add_task(
        self,
        name: str,
        prompt: str | None = None,
        model: str | None = None,
        handler: Callable[..., Awaitable[Any]] | None = None,
        depends_on: list[str] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 30.0,
        condition: Callable[[dict[str, TaskResult]], bool] | None = None,
        transform_input: Callable[[dict[str, TaskResult]], dict[str, Any]] | None = None,
    ) -> "Pipeline":
        """Add a task node to the pipeline. Returns self for chaining."""
        task = Task(
            name=name,
            prompt=prompt,
            model=model,
            handler=handler,
            depends_on=depends_on or [],
            priority=priority,
            timeout=timeout,
        )
        self._nodes[name] = PipelineNode(
            task=task,
            condition=condition,
            transform_input=transform_input,
        )
        return self

    def add_condition(
        self,
        name: str,
        predicate: Callable[[dict[str, TaskResult]], bool],
        if_true: str,
        if_false: str,
    ) -> "Pipeline":
        """Add conditional branching between two task paths."""
        # Create a routing task that evaluates the predicate
        async def condition_handler(task: Task, results: dict[str, TaskResult]) -> Any:
            return "true" if predicate(results) else "false"

        task = Task(name=name, handler=condition_handler)
        self._nodes[name] = PipelineNode(task=task)

        # Set conditions on the branch targets
        if if_true in self._nodes:
            self._nodes[if_true].condition = lambda r, n=name: (
                r.get(n) and r[n].output == "true"
            )
        if if_false in self._nodes:
            self._nodes[if_false].condition = lambda r, n=name: (
                r.get(n) and r[n].output == "false"
            )

        return self

    def set_error_handler(self, handler: Callable[[Exception], None]) -> "Pipeline":
        """Register a global error handling callback."""
        self._error_handler = handler
        return self

    @property
    def tasks(self) -> list[Task]:
        """Return all tasks in the pipeline."""
        return [node.task for node in self._nodes.values()]

    @property
    def task_names(self) -> list[str]:
        """Return all task names."""
        return list(self._nodes.keys())


class PipelineExecutor:
    """Executes pipelines using the model router and task scheduler."""

    def __init__(self, router: ModelRouter, max_concurrency: int = 10):
        self.router = router
        self.max_concurrency = max_concurrency

    async def execute(
        self,
        pipeline: Pipeline,
        input_data: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute a pipeline end-to-end."""
        start_time = time.perf_counter()
        input_data = input_data or {}

        logger.info(f"Starting pipeline: {pipeline.name} ({pipeline.id})")

        scheduler = TaskScheduler(max_concurrency=pipeline.config.max_concurrency)

        # Register all tasks with the scheduler
        for node in pipeline._nodes.values():
            scheduler.add_task(node.task)

        # Create the task executor that uses the model router
        async def task_executor(task: Task, results: dict[str, TaskResult]) -> TaskResult:
            return await self._execute_task(task, results, input_data, pipeline)

        try:
            results = await asyncio.wait_for(
                scheduler.execute_all(task_executor),
                timeout=pipeline.config.timeout,
            )
        except asyncio.TimeoutError:
            return PipelineResult(
                pipeline_id=pipeline.id,
                pipeline_name=pipeline.name,
                success=False,
                error=f"Pipeline timed out after {pipeline.config.timeout}s",
            )
        except Exception as e:
            if pipeline._error_handler:
                pipeline._error_handler(e)
            return PipelineResult(
                pipeline_id=pipeline.id,
                pipeline_name=pipeline.name,
                success=False,
                error=str(e),
            )

        # Aggregate results
        duration = (time.perf_counter() - start_time) * 1000
        metrics = self._compute_metrics(results, duration)

        # Get final output from the last completed task
        final_output = None
        for name in reversed(list(results.keys())):
            if results[name].status == TaskStatus.COMPLETED:
                final_output = results[name].output
                break

        success = all(r.status == TaskStatus.COMPLETED for r in results.values())

        return PipelineResult(
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            output=final_output,
            task_results=results,
            metrics=metrics,
            success=success,
        )

    async def _execute_task(
        self,
        task: Task,
        results: dict[str, TaskResult],
        input_data: dict[str, Any],
        pipeline: Pipeline,
    ) -> TaskResult:
        """Execute a single task, routing to appropriate model."""
        node = pipeline._nodes.get(task.name)

        # Check condition
        if node and node.condition and not node.condition(results):
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.CANCELLED,
                output=None,
            )

        # If task has a custom handler, use it
        if task.handler:
            output = await task.handler(task, results)
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                output=output,
            )

        # Otherwise, use the model router with the task's prompt
        if task.prompt:
            # Interpolate variables in the prompt
            prompt = self._interpolate_prompt(task.prompt, results, input_data)

            messages = [{"role": "user", "content": prompt}]
            response = await self.router.route(
                messages,
                preferred_provider=self._extract_provider(task.model),
                model=task.model,
            )

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                output=response.content,
                cost=response.cost,
                metadata={"model": response.model, "provider": response.provider},
            )

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            output=None,
        )

    def _interpolate_prompt(
        self,
        prompt: str,
        results: dict[str, TaskResult],
        input_data: dict[str, Any],
    ) -> str:
        """Replace {variable} placeholders in prompts."""
        # Replace {input} and {input.key} patterns
        for key, value in input_data.items():
            prompt = prompt.replace(f"{{input.{key}}}", str(value))
            prompt = prompt.replace("{input}", str(value))

        # Replace {task_name.output} patterns
        for name, result in results.items():
            prompt = prompt.replace(f"{{{name}.output}}", str(result.output or ""))
            prompt = prompt.replace(f"{{{name}}}", str(result.output or ""))

        return prompt

    @staticmethod
    def _extract_provider(model: str | None) -> str | None:
        """Extract provider hint from model string."""
        if not model:
            return None
        provider_hints = {
            "gpt": "openai",
            "claude": "anthropic",
            "gemini": "google",
        }
        for prefix, provider in provider_hints.items():
            if model.startswith(prefix):
                return provider
        return None

    @staticmethod
    def _compute_metrics(
        results: dict[str, TaskResult], duration_ms: float
    ) -> PipelineMetrics:
        """Compute aggregate metrics from task results."""
        metrics = PipelineMetrics(total_duration_ms=duration_ms)
        models_used = set()

        for result in results.values():
            if result.status == TaskStatus.COMPLETED:
                metrics.tasks_completed += 1
            elif result.status == TaskStatus.FAILED:
                metrics.tasks_failed += 1

            metrics.total_cost += result.cost
            if result.metadata.get("model"):
                models_used.add(result.metadata["model"])

        metrics.models_used = list(models_used)
        return metrics
