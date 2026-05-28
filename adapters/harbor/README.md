# Harbor Adapter

Agentic coding benchmark evaluation adapter for [eval-hub](https://github.com/eval-hub).

Runs [Harbor](https://github.com/harbor-ai/harbor) benchmark tasks on Kubernetes via EvalHub. Each task contains an instruction, test suite, and oracle solution. The adapter supports oracle verification (apply patches and run tests) and AI agent execution (Claude Code solving tasks autonomously).

## Architecture

```text
JobSpec → HarborAdapter
              ├── harbor CLI mode: `harbor run -p <task> -a <agent>`
              ├── kubernetes mode: K8s Job with task image
              └── import mode: parse existing result.json
          → parse reward from verifier output
          → map to JobResults with metrics
```

## Execution Modes

| Mode | Description |
|------|-------------|
| `kubernetes` | Run task as a K8s Job from a pre-built container image |
| `harbor` | Run via `harbor run` CLI (requires harbor installed) |
| `import` | Parse pre-existing results from a jobs directory |

## Configuration

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_path` | string | Path to Harbor task or dataset directory |
| `task_image` | string | Pre-built task container image (K8s mode) |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `execution_mode` | `kubernetes` | Execution mode |
| `agent` | `oracle` | Agent: `oracle`, `claude-code`, or `agent` |
| `model` | - | Model ID for AI agent mode |
| `namespace` | `evalhub` | K8s namespace |
| `timeout_sec` | `600` | Task timeout |
| `cpu` | `2` | CPU request/limit |
| `memory` | `4Gi` | Memory request/limit |
| `env_from_secrets` | `[]` | K8s Secrets to inject as env vars |
| `secret_volumes` | `[]` | Secret volume mounts |

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `reward` | benchmark | Per-task pass/fail (0 or 1) |
| `mean_reward` | benchmark | Average reward across tasks |
| `duration_seconds` | performance | Execution time |
| `cost_usd` | cost | API cost (agent mode) |
| `num_trials` | count | Number of tasks executed |
| `num_errored` | count | Number of failed tasks |

## RBAC Requirements

For Kubernetes execution mode, the adapter's service account needs:

```yaml
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list"]
```

## Local Development

```bash
cd adapters/harbor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-test.txt
.venv/bin/pytest tests/ -v

# Run with local job spec:
EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json .venv/bin/python main.py
```
