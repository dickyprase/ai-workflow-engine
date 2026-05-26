"""Core workflow engine with async task execution and lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from engine.config import EngineConfig
from engine.models import ModelRouter
from engine.pipeline import Pipeline, PipelineConfig, PipelineExecutor, PipelineResult
from engine.tasks import TaskDecomposer

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Main entry point for the AI Workflow Engine.

    Orchestrates pipelines, manages model routing, and provides
    a high-level API for workflow automation.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.router = ModelRouter(config)
        self.decomposer = TaskDecomposer()
        self._executor = PipelineExecutor(
            router=self.router,
            max_concurrency=config.max_concurrency,
        )
        self._pipelines: dict[str, Pipeline] = {}
        self._history: list[PipelineResult] = []
        logger.info("WorkflowEngine initialized")

    @classmethod
    def from_config(cls, path: str | Path) -> "WorkflowEngine":
        """Create a WorkflowEngine from a YAML/JSON configuration file."""
        config = EngineConfig.from_yaml(path)
        return cls(config)

    @classmethod
    def default(cls) -> "WorkflowEngine":
        """Create a WorkflowEngine with default configuration."""
        config = EngineConfig()
        return cls(config)

    def create_pipeline(
        self,
        name: str,
        description: str = "",
        max_concurrency: int | None = None,
        timeout: float | None = None,
    ) -> Pipeline:
        """Create and register a new pipeline."""
        pipeline_config = PipelineConfig(
            name=name,
            description=description,
            max_concurrency=max_concurrency or self.config.max_concurrency,
            timeout=timeout or self.config.default_timeout,
        )
        pipeline = Pipeline(pipeline_config)
        self._pipelines[name] = pipeline
        logger.info(f"Created pipeline: {name}")
        return pipeline

    async def run(
        self,
        pipeline: Pipeline | str,
        input: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Execute a pipeline with the given input data.

        Args:
            pipeline: Pipeline instance or registered pipeline name.
            input: Input data dictionary passed to the first task(s).
            **kwargs: Additional execution options.

        Returns:
            PipelineResult with output, metrics, and task-level results.
        """
        if isinstance(pipeline, str):
            if pipeline not in self._pipelines:
                raise ValueError(f"Pipeline '{pipeline}' not found. Register it first.")
            pipeline = self._pipelines[pipeline]

        logger.info(f"Running pipeline: {pipeline.name}")
        result = await self._executor.execute(pipeline, input_data=input)
        self._history.append(result)

        if result.success:
            logger.info(
                f"Pipeline '{pipeline.name}' completed in "
                f"{result.metrics.total_duration_ms:.1f}ms "
                f"(cost: ${result.metrics.total_cost:.4f})"
            )
        else:
            logger.error(f"Pipeline '{pipeline.name}' failed: {result.error}")

        return result

    async def run_batch(
        self,
        pipeline: Pipeline | str,
        inputs: list[dict[str, Any]],
        max_parallel: int = 5,
    ) -> list[PipelineResult]:
        """Execute a pipeline across multiple inputs in parallel.

        Args:
            pipeline: Pipeline instance or name.
            inputs: List of input dictionaries.
            max_parallel: Maximum parallel pipeline executions.

        Returns:
            List of PipelineResult objects.
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_one(input_data: dict[str, Any]) -> PipelineResult:
            async with semaphore:
                return await self.run(pipeline, input=input_data)

        results = await asyncio.gather(
            *[run_one(inp) for inp in inputs],
            return_exceptions=False,
        )
        return list(results)

    async def decompose_and_run(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        pipeline_name: str = "auto",
    ) -> PipelineResult:
        """Automatically decompose an objective into tasks and execute.

        Uses the TaskDecomposer to break down the objective, creates a
        pipeline from the resulting tasks, and executes it.
        """

        async def model_fn(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            response = await self.router.route(messages)
            return response.content

        tasks = await self.decomposer.decompose(
            objective=objective,
            context=context,
            model_fn=model_fn,
        )

        pipeline = self.create_pipeline(pipeline_name, description=objective)
        for task in tasks:
            pipeline.add_task(
                name=task.name,
                prompt=task.prompt,
                depends_on=task.depends_on,
                priority=task.priority,
            )

        return await self.run(pipeline)

    def get_pipeline(self, name: str) -> Pipeline | None:
        """Retrieve a registered pipeline by name."""
        return self._pipelines.get(name)

    @property
    def history(self) -> list[PipelineResult]:
        """Return execution history."""
        return self._history.copy()

    @property
    def total_cost(self) -> float:
        """Total cost across all pipeline executions."""
        return sum(r.metrics.total_cost for r in self._history)

    @property
    def stats(self) -> dict[str, Any]:
        """Engine-level statistics."""
        return {
            "pipelines_registered": len(self._pipelines),
            "total_runs": len(self._history),
            "successful_runs": sum(1 for r in self._history if r.success),
            "failed_runs": sum(1 for r in self._history if not r.success),
            "total_cost": self.total_cost,
            "available_providers": self.router.available_providers,
        }
