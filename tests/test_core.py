"""Tests for the AI Workflow Engine core functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.core import WorkflowEngine
from engine.config import EngineConfig, ModelProviderConfig, RoutingConfig
from engine.tasks import Task, TaskResult, TaskStatus, TaskScheduler, TaskDecomposer, TaskPriority
from engine.pipeline import Pipeline, PipelineConfig, PipelineExecutor


@pytest.fixture
def engine_config():
    """Create a test engine configuration."""
    return EngineConfig(
        max_concurrency=5,
        default_timeout=10.0,
        models={
            "openai": ModelProviderConfig(
                api_key="test-key",
                default_model="gpt-4o",
                max_tokens=1024,
            )
        },
        routing=RoutingConfig(
            strategy="cost_optimized",
            fallback_chain=["openai"],
        ),
    )


@pytest.fixture
def engine(engine_config):
    """Create a test workflow engine."""
    return WorkflowEngine(engine_config)


class TestWorkflowEngine:
    """Tests for the WorkflowEngine class."""

    def test_engine_creation(self, engine):
        assert engine is not None
        assert engine.config.max_concurrency == 5

    def test_engine_default(self):
        engine = WorkflowEngine.default()
        assert engine.config.max_concurrency == 10

    def test_create_pipeline(self, engine):
        pipeline = engine.create_pipeline("test-pipeline", description="A test")
        assert pipeline.name == "test-pipeline"
        assert pipeline.config.description == "A test"

    def test_get_pipeline(self, engine):
        engine.create_pipeline("my-pipe")
        result = engine.get_pipeline("my-pipe")
        assert result is not None
        assert result.name == "my-pipe"

    def test_get_nonexistent_pipeline(self, engine):
        assert engine.get_pipeline("nope") is None

    def test_stats_initial(self, engine):
        stats = engine.stats
        assert stats["pipelines_registered"] == 0
        assert stats["total_runs"] == 0
        assert stats["total_cost"] == 0.0


class TestPipeline:
    """Tests for Pipeline construction."""

    def test_add_task(self):
        config = PipelineConfig(name="test")
        pipeline = Pipeline(config)
        pipeline.add_task("step1", prompt="Do something")
        assert "step1" in pipeline.task_names

    def test_add_task_chaining(self):
        config = PipelineConfig(name="test")
        pipeline = Pipeline(config)
        result = pipeline.add_task("a").add_task("b").add_task("c")
        assert result is pipeline
        assert len(pipeline.task_names) == 3

    def test_task_dependencies(self):
        config = PipelineConfig(name="test")
        pipeline = Pipeline(config)
        pipeline.add_task("first", prompt="Step 1")
        pipeline.add_task("second", prompt="Step 2", depends_on=["first"])
        tasks = pipeline.tasks
        second = next(t for t in tasks if t.name == "second")
        assert "first" in second.depends_on


class TestTaskScheduler:
    """Tests for the DAG-based task scheduler."""

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        scheduler = TaskScheduler(max_concurrency=5)
        task = Task(name="simple", prompt="hello")
        scheduler.add_task(task)

        async def executor(t, results):
            return TaskResult(task_id=t.id, status=TaskStatus.COMPLETED, output="done")

        results = await scheduler.execute_all(executor)
        assert "simple" in results
        assert results["simple"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dependency_ordering(self):
        scheduler = TaskScheduler(max_concurrency=2)
        execution_order = []

        task_a = Task(name="a")
        task_b = Task(name="b", depends_on=["a"])
        task_c = Task(name="c", depends_on=["b"])

        scheduler.add_tasks([task_a, task_b, task_c])

        async def executor(t, results):
            execution_order.append(t.name)
            await asyncio.sleep(0.01)
            return TaskResult(task_id=t.id, status=TaskStatus.COMPLETED, output=t.name)

        await scheduler.execute_all(executor)
        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        scheduler = TaskScheduler(max_concurrency=5)

        # Three independent tasks should run in parallel
        for name in ["x", "y", "z"]:
            scheduler.add_task(Task(name=name))

        timestamps = {}

        async def executor(t, results):
            import time
            timestamps[t.name] = time.perf_counter()
            await asyncio.sleep(0.05)
            return TaskResult(task_id=t.id, status=TaskStatus.COMPLETED, output=t.name)

        await scheduler.execute_all(executor)

        # All should start within a small window (parallel)
        times = list(timestamps.values())
        assert max(times) - min(times) < 0.03  # Started nearly simultaneously

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self):
        scheduler = TaskScheduler()
        scheduler.add_task(Task(name="a", depends_on=["b"]))
        scheduler.add_task(Task(name="b", depends_on=["a"]))

        async def executor(t, results):
            return TaskResult(task_id=t.id, status=TaskStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="Circular dependency"):
            await scheduler.execute_all(executor)


class TestTaskDecomposer:
    """Tests for task decomposition."""

    @pytest.mark.asyncio
    async def test_rule_based_decompose_semicolons(self):
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("fetch data; process data; save results")
        assert len(tasks) == 3
        assert tasks[0].name == "step_1"
        assert tasks[1].depends_on == ["step_1"]
        assert tasks[2].depends_on == ["step_2"]

    @pytest.mark.asyncio
    async def test_rule_based_decompose_then(self):
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("read file then parse content then output")
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_single_task_no_split(self):
        decomposer = TaskDecomposer()
        tasks = await decomposer.decompose("just do this one thing")
        assert len(tasks) == 1
        assert tasks[0].name == "main"

    @pytest.mark.asyncio
    async def test_ai_decompose_with_mock(self):
        decomposer = TaskDecomposer()
        import json

        mock_response = json.dumps([
            {"name": "analyze", "description": "Analyze input", "depends_on": [], "priority": 2},
            {"name": "transform", "description": "Transform data", "depends_on": ["analyze"], "priority": 2},
        ])

        async def mock_model(prompt):
            return mock_response

        tasks = await decomposer.decompose("complex objective", model_fn=mock_model)
        assert len(tasks) == 2
        assert tasks[0].name == "analyze"
        assert tasks[1].depends_on == ["analyze"]


class TestEngineConfig:
    """Tests for configuration management."""

    def test_default_config(self):
        config = EngineConfig()
        assert config.max_concurrency == 10
        assert config.default_timeout == 30.0

    def test_from_dict(self):
        data = {
            "max_concurrency": 20,
            "default_timeout": 60.0,
            "models": {
                "openai": {
                    "api_key": "sk-test",
                    "default_model": "gpt-4o",
                }
            },
        }
        config = EngineConfig.from_dict(data)
        assert config.max_concurrency == 20
        assert config.models["openai"].api_key == "sk-test"

    def test_env_var_interpolation(self):
        import os
        os.environ["TEST_API_KEY"] = "secret-123"
        result = EngineConfig._interpolate_env_vars("key: ${TEST_API_KEY}")
        assert result == "key: secret-123"
        del os.environ["TEST_API_KEY"]

    def test_env_var_with_default(self):
        result = EngineConfig._interpolate_env_vars("key: ${MISSING_VAR:-fallback}")
        assert result == "key: fallback"
