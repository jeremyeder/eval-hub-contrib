"""Tests for the Harbor adapter.

Verifies results parsing, K8s job helpers, and adapter plumbing by
mocking subprocess and kubernetes calls with canned data.
"""

import json
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from evalhub.adapter import JobCallbacks, JobPhase, JobSpec, JobStatus, ModelConfig


def _make_spec(**overrides):
    """Create a valid JobSpec with sensible defaults."""
    defaults = {
        "id": "test-job-001",
        "provider_id": "harbor",
        "benchmark_id": "harbor-task",
        "benchmark_index": 0,
        "model": ModelConfig(url="local://test", name="claude-sonnet-4-6"),
        "parameters": {},
        "callback_url": "http://localhost:8080",
    }
    defaults.update(overrides)
    return JobSpec(**defaults)


# ---------------------------------------------------------------------------
# Phase 2: Results parser
# ---------------------------------------------------------------------------

class TestParseJob:
    def test_single_trial(self, tmp_path):
        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "test-001",
            "trials": [{
                "task_name": "task-0001",
                "status": "completed",
                "reward": 1.0,
                "metrics": {"duration_s": 42.5, "cost_usd": 0.03},
            }],
        }))

        from main import parse_job

        result = parse_job(tmp_path)
        assert result["job_id"] == "test-001"
        assert result["mean_reward"] == 1.0
        assert result["n_completed"] == 1
        assert result["n_errored"] == 0
        assert len(result["trials"]) == 1

    def test_multiple_trials(self, tmp_path):
        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "test-002",
            "trials": [
                {"task_name": "t1", "status": "completed", "reward": 1.0, "metrics": {}},
                {"task_name": "t2", "status": "completed", "reward": 0.0, "metrics": {}},
                {"task_name": "t3", "status": "completed", "reward": 1.0, "metrics": {}},
            ],
        }))

        from main import parse_job

        result = parse_job(tmp_path)
        assert result["n_completed"] == 3
        assert result["mean_reward"] == pytest.approx(2.0 / 3.0)

    def test_no_result_file(self, tmp_path):
        from main import parse_job

        with pytest.raises(FileNotFoundError):
            parse_job(tmp_path)

    def test_errored_trials(self, tmp_path):
        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "test-003",
            "trials": [
                {"task_name": "t1", "status": "completed", "reward": 1.0, "metrics": {}},
                {"task_name": "t2", "status": "errored", "reward": None, "metrics": {}},
            ],
        }))

        from main import parse_job

        result = parse_job(tmp_path)
        assert result["n_completed"] == 1
        assert result["n_errored"] == 1
        assert result["mean_reward"] == 1.0

    def test_all_errored(self, tmp_path):
        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "test-004",
            "trials": [
                {"task_name": "t1", "status": "errored", "reward": None, "metrics": {}},
            ],
        }))

        from main import parse_job

        result = parse_job(tmp_path)
        assert result["mean_reward"] is None
        assert result["n_errored"] == 1

    def test_metrics_are_evaluation_results(self, tmp_path):
        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "test-005",
            "trials": [{
                "task_name": "t1",
                "status": "completed",
                "reward": 1.0,
                "metrics": {"cost_usd": 0.03, "duration_s": 42.5},
            }],
        }))

        from main import parse_job

        trial = parse_job(tmp_path)["trials"][0]
        for m in trial["metrics"]:
            assert hasattr(m, "metric_name")
            assert hasattr(m, "metric_value")

        metric_map = {m.metric_name: m for m in trial["metrics"]}
        assert metric_map["cost_usd"].metric_value == 0.03
        assert metric_map["reward"].metric_value == 1.0


# ---------------------------------------------------------------------------
# Phase 3: K8s runner helpers
# ---------------------------------------------------------------------------

class TestScriptGeneration:
    def test_oracle_script(self):
        from main import _oracle_script

        script = _oracle_script()
        assert "solve.sh" in script
        assert "test.sh" in script
        assert "HARBOR_REWARD" in script

    def test_agent_script_with_model(self):
        from main import _agent_script

        script = _agent_script(model="claude-sonnet-4-6")
        assert "--model" in script
        assert "claude-sonnet-4-6" in script

    def test_agent_script_no_model(self):
        from main import _agent_script

        script = _agent_script(model="")
        assert "--model" not in script


class TestBuildVolumes:
    def test_empty(self):
        from main import _build_volumes

        volumes, mounts = _build_volumes(None)
        assert volumes is None
        assert mounts is None

    @patch("main._K8S_AVAILABLE", True)
    @patch("main.k8s_client")
    def test_with_secrets(self, mock_client):
        mock_client.V1KeyToPath = lambda key, path: types.SimpleNamespace(key=key, path=path)
        mock_client.V1Volume = lambda name, secret: types.SimpleNamespace(name=name, secret=secret)
        mock_client.V1SecretVolumeSource = lambda secret_name, items=None: types.SimpleNamespace(
            secret_name=secret_name, items=items)
        mock_client.V1VolumeMount = lambda name, mount_path, read_only: types.SimpleNamespace(
            name=name, mount_path=mount_path, read_only=read_only)

        from main import _build_volumes

        volumes, mounts = _build_volumes([
            {"secret_name": "api-keys", "mount_path": "/secrets/api-keys"},
            {"secret_name": "certs", "mount_path": "/secrets/certs",
             "items": [{"key": "ca.crt", "path": "ca.crt"}]},
        ])
        assert len(volumes) == 2
        assert len(mounts) == 2
        assert mounts[0].mount_path == "/secrets/api-keys"
        assert mounts[0].read_only is True


class TestRunTaskJob:
    @patch("main._load_k8s_config")
    @patch("main.k8s_client")
    def test_oracle_success(self, mock_client, mock_load):
        from main import run_task_job

        mock_batch = MagicMock()
        mock_core = MagicMock()
        mock_client.BatchV1Api.return_value = mock_batch
        mock_client.CoreV1Api.return_value = mock_core

        mock_status = MagicMock()
        mock_status.status.succeeded = True
        mock_status.status.failed = None
        mock_batch.read_namespaced_job_status.return_value = mock_status

        mock_pod = MagicMock()
        mock_pod.metadata.name = "harbor-test-pod"
        mock_core.list_namespaced_pod.return_value = MagicMock(items=[mock_pod])
        mock_core.read_namespaced_pod_log.return_value = "Running...\nHARBOR_REWARD=1.0\n"

        for attr in ("V1Job", "V1ObjectMeta", "V1JobSpec", "V1PodTemplateSpec",
                      "V1PodSpec", "V1PodSecurityContext", "V1Container",
                      "V1ResourceRequirements"):
            setattr(mock_client, attr, MagicMock())

        result = run_task_job(
            task_name="test", task_image="img:latest",
            namespace="evalhub", timeout_sec=300, agent="oracle",
        )

        assert result["reward"] == 1.0
        assert result["exit_code"] == 0
        mock_batch.create_namespaced_job.assert_called_once()

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            with patch("main._load_k8s_config"):
                with patch("main.k8s_client"):
                    from main import run_task_job
                    run_task_job(
                        task_name="t", task_image="i",
                        namespace="ns", agent="bad",
                    )


# ---------------------------------------------------------------------------
# Phase 4: HarborAdapter
# ---------------------------------------------------------------------------

class TestHarborAdapterImport:
    @patch("main._framework_adapter_init")
    def test_import_from_jobs_dir(self, mock_init, tmp_path):
        from main import HarborAdapter

        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "import-001",
            "trials": [
                {"task_name": "t1", "status": "completed", "reward": 1.0, "metrics": {}},
                {"task_name": "t2", "status": "completed", "reward": 0.0, "metrics": {}},
            ],
        }))

        spec = _make_spec()
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(jobs_dir=str(tmp_path))
        results = adapter.run_benchmark_job(spec, callbacks)

        assert results.overall_score == pytest.approx(0.5)
        assert results.num_examples_evaluated == 2

    @patch("main._framework_adapter_init")
    def test_import_no_result_json(self, mock_init, tmp_path):
        from main import HarborAdapter

        spec = _make_spec(id="job-002")
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(jobs_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            adapter.run_benchmark_job(spec, callbacks)


class TestHarborAdapterCli:
    @patch("main._framework_adapter_init")
    @patch("main.subprocess")
    @patch("main.tempfile")
    def test_harbor_cli_success(self, mock_tempfile, mock_subprocess, mock_init, tmp_path):
        from main import HarborAdapter

        mock_tempfile.mkdtemp.return_value = str(tmp_path)
        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")

        job_dir = tmp_path / "evalhub-test-job"
        job_dir.mkdir()
        (job_dir / "result.json").write_text(json.dumps({
            "job_id": "cli-001",
            "trials": [{"task_name": "t1", "status": "completed", "reward": 1.0, "metrics": {}}],
        }))

        spec = _make_spec(id="test-job-001", parameters={"agent": "oracle"})
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(task_path="/tasks/test", execution_mode="harbor")
        results = adapter.run_benchmark_job(spec, callbacks)

        assert results.overall_score == 1.0
        cmd = mock_subprocess.run.call_args[0][0]
        assert cmd[0] == "harbor"

    @patch("main._framework_adapter_init")
    @patch("main.subprocess")
    @patch("main.tempfile")
    def test_harbor_cli_failure(self, mock_tempfile, mock_subprocess, mock_init, tmp_path):
        from main import HarborAdapter

        mock_tempfile.mkdtemp.return_value = str(tmp_path)
        mock_subprocess.run.return_value = MagicMock(returncode=1, stderr="not found")

        spec = _make_spec(id="job-fail")
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(task_path="/tasks/missing", execution_mode="harbor")
        results = adapter.run_benchmark_job(spec, callbacks)

        assert results.overall_score == 0.0


class TestHarborAdapterK8s:
    @patch("main._framework_adapter_init")
    @patch("main.run_task_job")
    def test_k8s_dispatch(self, mock_run_task, mock_init):
        from main import HarborAdapter

        mock_run_task.return_value = {
            "reward": 1.0, "stdout": "HARBOR_REWARD=1.0\n",
            "duration_s": 120.0, "exit_code": 0,
        }

        spec = _make_spec(
            id="job-k8s",
            parameters={
                "execution_mode": "kubernetes",
                "task_image": "registry/task:latest",
                "task_path": "tasks/test",
                "agent": "oracle",
            },
        )
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(execution_mode="kubernetes")
        results = adapter.run_benchmark_job(spec, callbacks)

        mock_run_task.assert_called_once()
        assert results.overall_score == 1.0


class TestSecurityConstraints:
    def test_run_as_user_zero_rejected(self):
        from main import run_task_job

        with pytest.raises(ValueError, match="run_as_user must be >= 1"):
            with patch("main._load_k8s_config"):
                with patch("main.k8s_client") as mock_client:
                    for attr in ("V1Job", "V1ObjectMeta", "V1JobSpec",
                                 "V1PodTemplateSpec", "V1PodSpec",
                                 "V1PodSecurityContext", "V1Container",
                                 "V1ResourceRequirements", "V1SecurityContext",
                                 "V1Capabilities", "BatchV1Api", "CoreV1Api"):
                        setattr(mock_client, attr, MagicMock())
                    run_task_job(
                        task_name="t", task_image="i", namespace="evalhub",
                        agent="oracle", run_as_user=0,
                    )

    @patch("main._framework_adapter_init")
    @patch("main.run_task_job")
    def test_agent_mode_blocks_secret_mounts(self, mock_run_task, mock_init):
        from main import HarborAdapter

        spec = _make_spec(parameters={
            "execution_mode": "kubernetes",
            "task_image": "registry/task:latest",
            "task_path": "tasks/test",
            "agent": "claude-code",
            "env_from_secrets": ["api-keys"],
        })
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(execution_mode="kubernetes")
        with pytest.raises(ValueError, match="Agent mode cannot be combined with secret"):
            adapter.run_benchmark_job(spec, callbacks)

    def test_scrub_stdout_strips_non_reward_lines(self):
        from main import _scrub_stdout

        raw = "SECRET_KEY=abc123\nHARBOR_REWARD=1.0\nsome debug output\n"
        assert _scrub_stdout(raw) == "HARBOR_REWARD=1.0"


class TestStatusCallbacks:
    @patch("main._framework_adapter_init")
    def test_lifecycle_phases(self, mock_init, tmp_path):
        from main import HarborAdapter

        (tmp_path / "result.json").write_text(json.dumps({
            "job_id": "status-test",
            "trials": [{"task_name": "t1", "status": "completed", "reward": 1.0, "metrics": {}}],
        }))

        spec = _make_spec(id="job-status")
        callbacks = create_autospec(JobCallbacks)

        adapter = HarborAdapter(jobs_dir=str(tmp_path))
        adapter.run_benchmark_job(spec, callbacks)

        assert callbacks.report_status.call_count >= 2
        first = callbacks.report_status.call_args_list[0][0][0]
        assert first.status == JobStatus.RUNNING
        last = callbacks.report_status.call_args_list[-1][0][0]
        assert last.status == JobStatus.COMPLETED
