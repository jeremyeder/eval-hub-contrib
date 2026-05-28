"""Harbor benchmark adapter for eval-hub.

Runs Harbor agentic coding benchmark tasks via EvalHub. Supports three
execution modes:
  - harbor: runs `harbor run` CLI and parses results
  - kubernetes: runs tasks as K8s Jobs with oracle or agent modes
  - import: parses pre-existing results from a jobs directory

Architecture:
    1. JobSpec loaded from mounted ConfigMap (k8s mode) or local file
    2. Execution mode selected (harbor CLI, kubernetes, or import)
    3. Tasks executed and reward extracted from verifier output
    4. Results parsed into EvaluationResult metrics
    5. Structured JobResults returned to eval-hub service

Example usage:
    # In Kubernetes (automatic):
    python main.py  # Reads /meta/job.json

    # Local development:
    EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    DefaultCallbacks,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
)

try:
    from kubernetes import client as k8s_client, config as k8s_config
    from kubernetes.client.rest import ApiException
    _K8S_AVAILABLE = True
except ImportError:
    _K8S_AVAILABLE = False
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    ApiException = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security constraints
# ---------------------------------------------------------------------------

ALLOWED_NAMESPACES = frozenset(os.environ.get(
    "HARBOR_ALLOWED_NAMESPACES", "evalhub").split(","))

ALLOWED_IMAGE_PREFIXES = tuple(filter(None, os.environ.get(
    "HARBOR_ALLOWED_IMAGE_PREFIXES", "").split(","))) or None

ALLOWED_SECRET_PREFIXES = tuple(filter(None, os.environ.get(
    "HARBOR_ALLOWED_SECRET_PREFIXES", "").split(","))) or None

MAX_STDOUT_BYTES = 10_000


def _validate_namespace(namespace: str) -> None:
    if namespace not in ALLOWED_NAMESPACES:
        raise ValueError(
            f"Namespace {namespace!r} not in allowed list: "
            f"{sorted(ALLOWED_NAMESPACES)}")


def _validate_image(image: str) -> None:
    if ALLOWED_IMAGE_PREFIXES and not image.startswith(ALLOWED_IMAGE_PREFIXES):
        raise ValueError(
            f"Image {image!r} does not match any allowed prefix: "
            f"{ALLOWED_IMAGE_PREFIXES}")


def _validate_secret_names(names: list[str], context: str) -> None:
    if not ALLOWED_SECRET_PREFIXES:
        return
    for name in names:
        if not name.startswith(ALLOWED_SECRET_PREFIXES):
            raise ValueError(
                f"{context} secret {name!r} does not match any allowed "
                f"prefix: {ALLOWED_SECRET_PREFIXES}")


def _scrub_stdout(raw: str) -> str:
    """Extract only HARBOR_REWARD line from stdout, discard everything else."""
    lines = []
    for line in raw.splitlines():
        if line.startswith("HARBOR_REWARD="):
            lines.append(line)
    return "\n".join(lines)[:MAX_STDOUT_BYTES]


# ---------------------------------------------------------------------------
# Results parser
# ---------------------------------------------------------------------------

_METRIC_TYPES = {
    "reward": "benchmark",
    "mean_reward": "benchmark",
    "duration_s": "performance",
    "cost_usd": "cost",
    "input_tokens": "count",
    "output_tokens": "count",
    "env_build_seconds": "performance",
    "agent_exec_seconds": "performance",
    "verifier_seconds": "performance",
}


def parse_job(job_dir: Path) -> dict:
    """Parse a Harbor job directory and return structured results.

    Returns:
        Dict with keys: job_id, trials, mean_reward, n_completed, n_errored.
    """
    result_path = job_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"No result.json found in {job_dir}")

    with open(result_path) as f:
        raw = json.load(f)

    trials = []
    completed_rewards = []
    n_errored = 0

    for raw_trial in raw.get("trials", []):
        status = raw_trial.get("status", "completed")

        if status == "errored":
            n_errored += 1
            trials.append({"task_name": raw_trial["task_name"], "metrics": []})
            continue

        reward = raw_trial.get("reward")
        if reward is not None:
            completed_rewards.append(reward)

        metrics = [EvaluationResult(
            metric_name="reward",
            metric_value=reward if reward is not None else 0.0,
            metric_type="benchmark",
        )]

        for key, value in raw_trial.get("metrics", {}).items():
            if value is not None:
                metrics.append(EvaluationResult(
                    metric_name=key,
                    metric_value=value,
                    metric_type=_METRIC_TYPES.get(key, "float"),
                ))

        trials.append({"task_name": raw_trial["task_name"], "metrics": metrics})

    mean_reward = (
        sum(completed_rewards) / len(completed_rewards)
        if completed_rewards else None
    )

    return {
        "job_id": raw.get("job_id", ""),
        "trials": trials,
        "mean_reward": mean_reward,
        "n_completed": len(completed_rewards),
        "n_errored": n_errored,
    }


# ---------------------------------------------------------------------------
# K8s runner helpers
# ---------------------------------------------------------------------------

def _require_k8s():
    if not _K8S_AVAILABLE:
        raise RuntimeError(
            "kubernetes package required. Install with: pip install kubernetes>=29.0"
        )


def _load_k8s_config():
    _require_k8s()
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()


def _oracle_script() -> str:
    return """#!/bin/bash
set -o pipefail
cd /app
mkdir -p /logs/verifier
bash /solution/solve.sh
bash /tests/test.sh
echo "HARBOR_REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo 0)"
"""


def _agent_script(model: str = "") -> str:
    model_flag = f"--model {shlex.quote(model)}" if model else ""
    return f"""#!/bin/bash
set -o pipefail
cd /app
mkdir -p /logs/verifier

claude -p --dangerously-skip-permissions {model_flag} \
  "Read /app/instruction.md and implement the solution in this codebase. \
You may run tests to verify your work and iterate until tests pass. \
Do not modify files under /tests/."

bash /tests/test.sh
echo "HARBOR_REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo 0)"
"""


def _build_volumes(
    secret_volumes: list[dict] | None,
) -> tuple[list | None, list | None]:
    if not secret_volumes:
        return None, None
    volumes = []
    volume_mounts = []
    for sv in secret_volumes:
        vol_name = f"secret-{sv['secret_name']}"
        items = None
        if sv.get("items"):
            items = [
                k8s_client.V1KeyToPath(key=item["key"], path=item["path"])
                for item in sv["items"]
            ]
        volumes.append(k8s_client.V1Volume(
            name=vol_name,
            secret=k8s_client.V1SecretVolumeSource(
                secret_name=sv["secret_name"], items=items),
        ))
        volume_mounts.append(k8s_client.V1VolumeMount(
            name=vol_name, mount_path=sv["mount_path"], read_only=True,
        ))
    return volumes, volume_mounts


def run_task_job(
    task_name: str,
    task_image: str,
    namespace: str,
    timeout_sec: int = 600,
    cpu: str = "2",
    memory: str = "4Gi",
    run_as_user: int = 1001,
    env_from_secrets: list[str] | None = None,
    env_from_configmaps: list[str] | None = None,
    agent: str = "oracle",
    model: str = "",
    secret_volumes: list[dict] | None = None,
) -> dict[str, Any]:
    """Run a Harbor task as a K8s Job and return the result."""
    _load_k8s_config()
    batch_v1 = k8s_client.BatchV1Api()
    core_v1 = k8s_client.CoreV1Api()

    job_name = f"harbor-{task_name}-{int(time.monotonic()) % 100000}"

    if agent == "oracle":
        script = _oracle_script()
    elif agent in ("claude-code", "agent"):
        script = _agent_script(model=model)
    else:
        raise ValueError(
            f"Unknown agent: {agent!r}. Must be 'oracle', 'claude-code', or 'agent'.")

    volumes, volume_mounts = _build_volumes(secret_volumes)

    if run_as_user < 1:
        raise ValueError(
            f"run_as_user must be >= 1 (got {run_as_user}). "
            "Running as root (UID 0) is not allowed.")

    job = k8s_client.V1Job(
        metadata=k8s_client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=k8s_client.V1JobSpec(
            backoff_limit=0,
            active_deadline_seconds=timeout_sec,
            template=k8s_client.V1PodTemplateSpec(
                spec=k8s_client.V1PodSpec(
                    restart_policy="Never",
                    automount_service_account_token=False,
                    enable_service_links=False,
                    security_context=k8s_client.V1PodSecurityContext(
                        run_as_user=run_as_user,
                        run_as_non_root=True,
                    ),
                    volumes=volumes,
                    containers=[k8s_client.V1Container(
                        name="task",
                        image=task_image,
                        image_pull_policy="Always",
                        command=["/bin/bash", "-c", script],
                        resources=k8s_client.V1ResourceRequirements(
                            requests={"cpu": cpu, "memory": memory},
                            limits={"cpu": cpu, "memory": memory},
                        ),
                        security_context=k8s_client.V1SecurityContext(
                            allow_privilege_escalation=False,
                            capabilities=k8s_client.V1Capabilities(
                                drop=["ALL"]),
                        ),
                        env_from=[
                            *[k8s_client.V1EnvFromSource(
                                secret_ref=k8s_client.V1SecretEnvSource(name=s))
                              for s in (env_from_secrets or [])],
                            *[k8s_client.V1EnvFromSource(
                                config_map_ref=k8s_client.V1ConfigMapEnvSource(name=c))
                              for c in (env_from_configmaps or [])],
                        ] or None,
                        volume_mounts=volume_mounts,
                    )],
                ),
            ),
        ),
    )

    start_time = time.monotonic()
    logger.info("Creating K8s Job %s in %s", job_name, namespace)
    batch_v1.create_namespaced_job(namespace, job)

    reward = 0.0
    stdout = ""
    exit_code = -1

    try:
        while time.monotonic() - start_time < timeout_sec + 30:
            job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
            if job_status.status.succeeded:
                exit_code = 0
                break
            if job_status.status.failed:
                exit_code = 1
                break
            time.sleep(5)

        pods = core_v1.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}")
        if pods.items:
            pod_name = pods.items[0].metadata.name
            try:
                stdout = core_v1.read_namespaced_pod_log(pod_name, namespace)
                for line in stdout.splitlines():
                    if line.startswith("HARBOR_REWARD="):
                        try:
                            reward = float(line.split("=", 1)[1].strip())
                        except (ValueError, TypeError):
                            logger.warning("Malformed reward line: %s", line)
            except ApiException as e:
                logger.warning("Failed to read pod logs: %s", e)
    finally:
        try:
            batch_v1.delete_namespaced_job(
                job_name, namespace, propagation_policy="Background")
        except ApiException:
            pass

    return {
        "reward": reward,
        "stdout": stdout,
        "duration_s": round(time.monotonic() - start_time, 1),
        "exit_code": exit_code,
    }


# ---------------------------------------------------------------------------
# EvalHub adapter
# ---------------------------------------------------------------------------

def _framework_adapter_init(adapter_instance):
    FrameworkAdapter.__init__(adapter_instance)


def _build_job_results(
    config: JobSpec,
    job_data: dict,
    agent_name: str,
    model_name: str,
    task_path: str,
    duration_s: float,
) -> JobResults:
    """Map parsed Harbor job data to EvalHub JobResults."""
    trials = job_data["trials"]
    all_metrics = []

    mean_reward = job_data["mean_reward"]
    if mean_reward is None:
        logger.warning("mean_reward is None — no trials completed successfully")
        mean_reward = 0.0

    all_metrics.append(EvaluationResult(
        metric_name="mean_reward", metric_value=mean_reward, metric_type="benchmark"))
    all_metrics.append(EvaluationResult(
        metric_name="num_trials", metric_value=len(trials), metric_type="count"))
    all_metrics.append(EvaluationResult(
        metric_name="num_errored", metric_value=job_data["n_errored"], metric_type="count"))

    for trial in trials:
        prefix = trial["task_name"].replace("/", "_")
        for metric in trial["metrics"]:
            all_metrics.append(EvaluationResult(
                metric_name=f"{prefix}/{metric.metric_name}",
                metric_value=metric.metric_value,
                metric_type=metric.metric_type,
            ))

    evaluation_metadata = {
        "harbor_job_id": job_data["job_id"],
        "agent": agent_name,
        "task_path": task_path,
        "n_completed": job_data["n_completed"],
        "n_errored": job_data["n_errored"],
    }
    if model_name:
        evaluation_metadata["model"] = model_name

    total_cost = sum(
        m.metric_value for t in trials for m in t["metrics"]
        if m.metric_name == "cost_usd" and m.metric_value is not None
    )
    if total_cost:
        all_metrics.append(EvaluationResult(
            metric_name="total_cost_usd", metric_value=total_cost, metric_type="cost"))

    return JobResults(
        id=config.id,
        benchmark_id=config.benchmark_id,
        benchmark_index=config.benchmark_index,
        model_name=model_name or agent_name,
        results=all_metrics,
        overall_score=job_data["mean_reward"],
        num_examples_evaluated=len(trials),
        duration_seconds=duration_s,
        completed_at=datetime.now(timezone.utc),
        evaluation_metadata=evaluation_metadata,
    )


class HarborAdapter(FrameworkAdapter):
    """Harbor benchmark adapter for eval-hub.

    Runs Harbor agentic coding benchmark tasks and maps results to
    EvalHub JobResults. Supports harbor CLI, Kubernetes, and import modes.

    Args:
        task_path: Path to Harbor task or dataset directory.
        jobs_dir: Path to pre-existing Harbor results (import mode).
        execution_mode: Override execution mode (harbor, kubernetes, import).
        job_spec_path: Path to job.json for standalone execution.
    """

    def __init__(
        self,
        task_path: str | None = None,
        jobs_dir: str | None = None,
        execution_mode: str | None = None,
        job_spec_path: str | None = None,
    ):
        _framework_adapter_init(self)
        self._task_path = task_path
        self._jobs_dir = jobs_dir
        self._execution_mode = execution_mode
        self._job_spec_path = job_spec_path

    def run_benchmark_job(
        self, config: JobSpec, callbacks: JobCallbacks
    ) -> JobResults:
        start_time = time.monotonic()
        params = config.parameters or {}

        task_path = self._task_path or params.get("task_path", "")
        agent_name = params.get("agent", "oracle")
        model_name = (
            params.get("model")
            or (config.model.name if config.model else "")
            or ""
        )
        jobs_dir = self._jobs_dir or params.get("jobs_dir")
        execution_mode = (
            self._execution_mode or params.get("execution_mode", "harbor")
        )

        if jobs_dir:
            return self._import_results(
                config, callbacks, Path(jobs_dir),
                agent_name, model_name, task_path, start_time,
            )

        if execution_mode == "kubernetes":
            return self._run_k8s(
                config, callbacks,
                task_path, agent_name, model_name, params, start_time,
            )

        return self._run_harbor(
            config, callbacks,
            task_path, agent_name, model_name, params, start_time,
        )

    def _run_harbor(self, config, callbacks, task_path, agent_name,
                    model_name, params, start_time):
        if not task_path:
            raise ValueError("task_path is required")

        n_concurrent = int(params.get("n_concurrent", 1))
        timeout_multiplier = float(params.get("timeout_multiplier", 1.0))

        self._report_status(callbacks, JobStatus.RUNNING, JobPhase.INITIALIZING,
                            f"Preparing harbor run: {task_path} agent={agent_name}")

        jobs_dir = tempfile.mkdtemp(prefix="harbor-evalhub-")
        job_name = f"evalhub-{config.id[:8]}" if config.id else "evalhub-run"

        self._report_status(callbacks, JobStatus.RUNNING,
                            JobPhase.RUNNING_EVALUATION, f"Running harbor: {task_path}")

        cmd = [
            "harbor", "run", "-p", task_path, "-a", agent_name,
            "--jobs-dir", jobs_dir, "--job-name", job_name,
            "--n-concurrent", str(n_concurrent),
            "--timeout-multiplier", str(timeout_multiplier),
        ]
        if model_name:
            cmd.extend(["-m", model_name])

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            logger.error("harbor run failed (exit %d): %s",
                         result.returncode, result.stderr[:500])
            self._report_status(callbacks, JobStatus.FAILED,
                                JobPhase.RUNNING_EVALUATION,
                                f"harbor run failed with exit code {result.returncode}")
            return JobResults(
                id=config.id, benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=model_name or agent_name,
                results=[EvaluationResult(
                    metric_name="harbor_exit_code",
                    metric_value=result.returncode, metric_type="status")],
                overall_score=0.0, num_examples_evaluated=0,
                duration_seconds=time.monotonic() - start_time,
                completed_at=datetime.now(timezone.utc),
                evaluation_metadata={"exit_code": result.returncode},
            )

        job_dir = Path(jobs_dir) / job_name
        return self._parse_and_map(config, callbacks, job_dir,
                                   agent_name, model_name, task_path, start_time)

    def _run_k8s(self, config, callbacks, task_path, agent_name,
                 model_name, params, start_time):
        task_image = params.get("task_image", "")
        namespace = params.get("namespace", "evalhub")
        timeout = int(params.get("timeout_sec", 600))
        cpu = params.get("cpu", "2")
        memory = params.get("memory", "4Gi")
        run_as_user = 1001  # hardcoded — do not accept from params
        env_from_secrets = params.get("env_from_secrets", [])
        if isinstance(env_from_secrets, str):
            env_from_secrets = [env_from_secrets]
        env_from_configmaps = params.get("env_from_configmaps", [])
        if isinstance(env_from_configmaps, str):
            env_from_configmaps = [env_from_configmaps]
        secret_volumes = params.get("secret_volumes", [])

        if not task_image:
            raise ValueError("task_image is required for kubernetes execution mode")

        _validate_namespace(namespace)
        _validate_image(task_image)
        _validate_secret_names(env_from_secrets, "env_from_secrets")
        _validate_secret_names(
            [sv["secret_name"] for sv in secret_volumes if isinstance(sv, dict)],
            "secret_volumes")

        if agent_name in ("claude-code", "agent") and (env_from_secrets or secret_volumes):
            raise ValueError(
                "Agent mode cannot be combined with secret mounts — "
                "the agent runs with --dangerously-skip-permissions and "
                "could exfiltrate mounted secrets. Use oracle mode for "
                "tasks that need secrets, or remove secret mounts.")

        task_name = task_path.replace("/", "-").replace("tasks-", "")

        self._report_status(callbacks, JobStatus.RUNNING,
                            JobPhase.RUNNING_EVALUATION,
                            f"Running K8s Job: {task_name} (agent={agent_name})")

        result = run_task_job(
            task_name=task_name, task_image=task_image, namespace=namespace,
            timeout_sec=timeout, cpu=cpu, memory=memory,
            run_as_user=run_as_user, env_from_secrets=env_from_secrets,
            env_from_configmaps=env_from_configmaps, agent=agent_name,
            model=model_name, secret_volumes=secret_volumes,
        )

        all_metrics = [
            EvaluationResult(metric_name="reward",
                             metric_value=result["reward"], metric_type="benchmark"),
            EvaluationResult(metric_name="mean_reward",
                             metric_value=result["reward"], metric_type="benchmark"),
            EvaluationResult(metric_name="duration_seconds",
                             metric_value=result["duration_s"], metric_type="performance"),
            EvaluationResult(metric_name="num_trials",
                             metric_value=1, metric_type="count"),
        ]

        status = JobStatus.COMPLETED if result["exit_code"] == 0 else JobStatus.FAILED
        self._report_status(callbacks, status, JobPhase.COMPLETED,
                            f"K8s Job complete: reward={result['reward']}", progress=1.0)

        return JobResults(
            id=config.id, benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=model_name or agent_name,
            results=all_metrics, overall_score=result["reward"],
            num_examples_evaluated=1,
            duration_seconds=time.monotonic() - start_time,
            completed_at=datetime.now(timezone.utc),
            evaluation_metadata={
                "agent": agent_name, "task_path": task_path,
                "task_image": task_image, "execution_mode": "kubernetes",
                "exit_code": result["exit_code"],
                "test_output": _scrub_stdout(result.get("stdout", "")),
            },
        )

    def _import_results(self, config, callbacks, jobs_dir, agent_name,
                        model_name, task_path, start_time):
        self._report_status(callbacks, JobStatus.RUNNING, JobPhase.LOADING_DATA,
                            f"Importing results from {jobs_dir}")

        if (jobs_dir / "result.json").exists():
            job_dir = jobs_dir
        else:
            subdirs = [
                d for d in sorted(jobs_dir.iterdir())
                if d.is_dir() and (d / "result.json").exists()
            ]
            if not subdirs:
                raise FileNotFoundError(f"No Harbor result.json found in {jobs_dir}")
            job_dir = subdirs[0]

        return self._parse_and_map(config, callbacks, job_dir,
                                   agent_name, model_name, task_path, start_time)

    def _parse_and_map(self, config, callbacks, job_dir, agent_name,
                       model_name, task_path, start_time):
        self._report_status(callbacks, JobStatus.RUNNING, JobPhase.POST_PROCESSING,
                            "Parsing harbor results")

        job_data = parse_job(job_dir)
        job_results = _build_job_results(
            config, job_data, agent_name, model_name, task_path,
            duration_s=time.monotonic() - start_time,
        )

        self._report_status(
            callbacks, JobStatus.COMPLETED, JobPhase.COMPLETED,
            f"Harbor benchmark complete: {len(job_data['trials'])} trials, "
            f"mean_reward={job_data['mean_reward']}", progress=1.0)

        return job_results

    @staticmethod
    def _report_status(callbacks, status, phase, message,
                       progress=None, total_steps=None, completed_steps=None):
        try:
            callbacks.report_status(JobStatusUpdate(
                status=status, phase=phase, progress=progress,
                message=MessageInfo(message=message, message_code="info"),
                total_steps=total_steps, completed_steps=completed_steps,
                timestamp=datetime.now(timezone.utc),
            ))
        except Exception as exc:
            logger.warning("Failed to report status: %s", exc)


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

def main():
    import os
    import sys
    import traceback

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "/meta/job.json"
    logger.info("Loading job spec from %s", config_path)

    try:
        spec = JobSpec.from_file(config_path)
    except FileNotFoundError:
        sys.exit(f"Job spec not found: {config_path}")
    except Exception as e:
        sys.exit(f"Failed to load job spec: {e}")

    logger.info("Job: id=%s benchmark=%s model=%s",
                spec.id, spec.benchmark_id,
                spec.model.name if spec.model else "none")

    task_path = os.environ.get("HARBOR_TASK_PATH", "")
    adapter = HarborAdapter(task_path=task_path or None)
    callbacks = DefaultCallbacks(job_id=spec.id, benchmark_id=spec.benchmark_id)

    try:
        results = adapter.run_benchmark_job(spec, callbacks)
    except Exception:
        logger.error("run_benchmark_job failed:\n%s", traceback.format_exc())
        sys.exit(1)

    if results is None:
        logger.error("run_benchmark_job returned None")
        sys.exit(1)

    try:
        rid = callbacks.mlflow.save(results, spec)
        if rid:
            results.mlflow_run_id = rid
            logger.info("MLflow run: %s", rid)
    except Exception as exc:
        logger.warning("MLflow save failed (non-fatal): %s", exc)

    callbacks.report_results(results)
    logger.info("Completed: %d examples, overall_score=%s",
                results.num_examples_evaluated, results.overall_score)


if __name__ == "__main__":
    main()
