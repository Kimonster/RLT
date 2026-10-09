# Codex 执行任务：RLT 第 1 步——16 维动作转换只读审计

> **请直接执行本任务，而不是再次给出操作建议。** 本轮只做动作转换的合成数据单元测试和相关源码只读检查。完成后生成报告并停止，不修复业务代码，不训练，不启动仿真，不进入下一步。本文已内嵌完整检查脚本，不需要其他附件，也不需要联网下载。

## 1. 任务背景与已知事实

用户正在 GenieSim 仿真中复现 RLT，任务是叠三个方块。本地仓库经过修改，必须以本地源码和实际测试为准，不能用公开仓库版本覆盖本地实现。

以下信息由用户提供，作为本次检查的预期规格；尚未通过本轮工具验证：

| 项目 | 已知值 |
|---|---|
| 仓库目录 | `/root/workspace/rlt/RLT` |
| Python 解释器 | `/root/workspace/envs/openpi/bin/python` |
| Stage-1 | `rlt_alpha=0.0`，VLA 冻结，仅训练 RL-token 模块 |
| Stage-2 特征版本 | replay 使用 Stage-1 **20k checkpoint**；后来虽续训到 200k，本轮不切换版本 |
| 真实动作维度 | 16 维：索引 `0:14` 是关节，`14:16` 是夹爪 |
| Chunk 长度 | 50 |
| 动作表示 | 前 14 维相对当前 chunk 起点的 proprio 做 delta；后 2 维保持 absolute；随后对动作做 quantile normalization |
| Replay | 150 个 episode、4,122 条 transition，全部来自 VLA；原始 journal 保存 absolute action |
| 本轮目标 | 验证合成数据上的动作表示、归一化、反变换和训练/推理入口是否一致 |

Stage-2 已落盘配置：

```text
/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2_retrain_conservative_8gpu/analysis/stage2_training_config.json
```

不要因为文件名、配置字段或 `action_chunk == ref_chunk` 就宣称动作转换正确。正向和反向转换若都错用相同维度，也可能通过 round-trip 测试。

## 2. 执行边界

允许读取本地配置、相关源码和 Git 状态；允许运行本文提供的 CPU 合成数据测试。所有新建脚本、日志和报告必须放在 `/tmp/rlt_step1_action_audit/` 下的本次独立目录中。不要将脚本写入业务仓库。

**禁止执行以下操作：** 修改或格式化仓库源码、修改配置或归一化统计量、修改 checkpoint/replay、执行训练或数据导出脚本、启动仿真或策略服务、占用 GPU、安装或升级依赖、联网拉取代码、执行 `git reset/checkout/restore/clean/pull`、删除已有文件。

如果存在未提交修改，保留原样。缺路径、缺依赖、导入失败、函数签名变化或测试失败时，记录具体错误并停止。不要为了得到 PASS 而改测试期望值、改容差、补装依赖、手工替换被测模块或 monkey-patch 配置。

## 3. 独立的预期定义

对 absolute action `A`、当前状态 `p`，本次审计使用的规格为：

```text
represented[..., 0:14] = A[..., 0:14] - p[..., 0:14]
represented[..., 14:16] = A[..., 14:16]
normalized = (represented - q01) / (q99 - q01 + 1e-6) * 2 - 1
```

`p` 沿 chunk 时间维广播。这里的 delta 是**相对 chunk 起点状态**，不是相邻动作之间作差。

夹爪保持 absolute 的意思是“不减当前夹爪状态”，不是“不做归一化”。`next_ref_chunk` 必须以 `next_proprio` 为基准。

本次使用 4 个样本、每个 50 步、每步 16 维；状态非零，且当前与下一状态不同。归一化统计量也是人工构造的，**不加载真实统计量**。这些输入没有零行 padding，因而本轮不覆盖 padding、有效长度 mask 或边界裁剪。

配置 JSON 仅用于记录背景字段；单元测试会显式构造 `action_dim=16`、`proprio_dim=16`、`chunk_len=50`、`delta_action_dims=14` 的测试配置。**这不是从实际训练脚本重建完整运行配置，不能证明配置字段在训练进程中已正确传递。**

## 4. 验收项目

| 测试名称 | 要验证的内容 |
|---|---|
| `train_ref_semantics_14d` | 训练参考动作符合前 14 维 delta 的独立定义 |
| `train_action_semantics_14d` | 训练实际动作符合相同定义 |
| `next_ref_uses_next_proprio` | 下一参考动作使用下一状态，而不是当前状态 |
| `train_vs_inference_normalization` | 批量训练与逐条推理的归一化结果一致 |
| `reference_roundtrip_numpy` | NumPy 正、反变换能还原 absolute action |
| `gripper_semantics_absolute_before_normalization` | 两维夹爪不减当前状态，但参与归一化 |
| `jax_inverse_semantics_14d` | 独立 JAX 反变换也符合 14 维定义 |

合成测试的最大绝对误差容差是 **`1e-5`**。这是本次数值测试的容差，不是机器人控制精度标准。JAX 测试若显式传入 `delta_action_dims=14`，只证明该函数在此参数下的行为；仍需区分它与真实调用点是否正确传参。

## 5. 执行命令（完整脚本已内嵌）

请在用户的执行环境中用 **Bash** 执行下面整段命令。它会创建独立工作目录、保存检查脚本、运行测试并记录日志。所有路径已经填写，不要求用户手工拼接命令。

如果命令因测试未通过而返回 `2`，这表示需要检查报告，不授权自动修复。执行后继续完成第 6 节的报告整理，然后停止。

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="/root/workspace/rlt/RLT"
PY="/root/workspace/envs/openpi/bin/python"
CONFIG="/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2_retrain_conservative_8gpu/analysis/stage2_training_config.json"
BASE="/tmp/rlt_step1_action_audit"

mkdir -p "$BASE"
WORK_DIR="$(mktemp -d "$BASE/codex_XXXXXXXX")"
printf 'WORK_DIR=%s\n' "$WORK_DIR"
printf '%s\n' "$WORK_DIR" > "$WORK_DIR/work_dir.txt"

# Avoid bytecode writes into the repo and optional Git index refreshes.
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$WORK_DIR/cache"
export TMPDIR="$WORK_DIR/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"

if [[ ! -d "$REPO" || ! -x "$PY" || ! -r "$CONFIG" ]]; then
  {
    echo 'OVERALL_STATUS: INCOMPLETE'
    echo 'Required local path is missing or inaccessible; no tests were run.'
    printf 'REPO=%s\nPY=%s\nCONFIG=%s\n' "$REPO" "$PY" "$CONFIG"
    [[ -d "$REPO" ]] && echo 'repo: present' || echo 'repo: missing'
    [[ -x "$PY" ]] && echo 'python: executable' || echo 'python: missing/not executable'
    [[ -r "$CONFIG" ]] && echo 'config: readable' || echo 'config: missing/not readable'
    echo 'Stop; do not install, clone, or replace files.'
  } | tee "$WORK_DIR/blocked.txt"
  exit 2
fi

cd "$REPO"
if command -v git >/dev/null 2>&1; then
  git -C "$REPO" status --porcelain=v1 --untracked-files=normal \
    > "$WORK_DIR/git_status_before.txt" 2> "$WORK_DIR/git_status_before.stderr" || true
else
  echo 'git unavailable; working tree comparison not performed' \
    > "$WORK_DIR/git_status_before.stderr"
fi

# Write only the independent audit script, not repository source files.
cat > "$WORK_DIR/rlt_step1_action_audit.py" <<'PY_RLT_ACTION_AUDIT'
#!/usr/bin/env python3
"""Read-only RLT action-adapter unit audit for the user's 16D GenieSim setup.

Runs the LOCAL adapter on SYNTHETIC inputs and synthetic quantile statistics.
Does not load checkpoints/replay, launch simulation, train, or edit source.
A pass does NOT certify live serving, real statistics, or replay timing.
Only writes a new report directory. CPU only. No network requests.

Public interface checked against:
https://github.com/Yyshadow/openpi-RLT/blob/main/rlt_online_rl/src/rlt_online_rl/action_representation.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

# Must be set before the local module imports JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

DEFAULT_REPO = "/root/workspace/rlt/RLT"
DEFAULT_CONFIG = (
    "/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/"
    "stage2_retrain_conservative_8gpu/analysis/stage2_training_config.json"
)
FIELDS = {
    "gamma", "action_dim", "proprio_dim", "chunk_len", "delta_action_dims",
    "action_representation", "action_norm_stats_path", "fixed_std",
    "actor_residual_scale", "reference_dropout_prob", "target_actor_deterministic",
    "bc_weight", "q_weight", "delta_weight", "actor_q_start_step",
    "online_bc_weight", "online_q_weight", "checkpoint_dir", "replay_path",
}


def config_entries(obj: object, prefix: str = "") -> dict:
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FIELDS:
                result[path] = value
            if isinstance(value, (dict, list)):
                result.update(config_entries(value, path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                result.update(config_entries(value, f"{prefix}[{i}]"))
    return result


def git_rev(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def collect_sources(repo: Path, out: Path) -> None:
    # Never execute the training or export scripts; read small, relevant excerpts.
    files = [
        "rlt_online_rl/src/rlt_online_rl/action_representation.py",
        "rlt_online_rl/src/rlt_online_rl/config.py",
        "rlt_online_rl/src/rlt_online_rl/networks.py",
        "rlt_online_rl/src/rlt_online_rl/trainer.py",
        "scripts/geniesim/train_stage2_plan.py",
        "scripts/geniesim/export_rollout_replay.py",
    ]
    pattern = re.compile(
        r"ActionRepresentationAdapter|prepare_training_batch|normalize_ref_chunk|"
        r"denormalize_to_abs_chunk|delta_action_dims|actor_residual_scale|"
        r"def actor_mean|def sample_action|tanh|gamma|action_norm_stats_path"
    )
    sections = ["LOCAL SOURCE EXCERPTS; not an executed-runtime call trace.\n"]
    for relative in files:
        path = repo / relative
        sections.append(f"\n=== {relative} ===\n")
        if not path.is_file():
            sections.append("MISSING\n")
            continue
        raw = path.read_bytes()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        sections.append(f"sha256: {hashlib.sha256(raw).hexdigest()}\n")
        if path.name == "action_representation.py":
            indices = list(range(min(len(lines), 650)))
        else:
            selected = set()
            for i, line in enumerate(lines):
                if pattern.search(line):
                    selected.update(range(max(0, i - 4), min(len(lines), i + 10)))
            indices = sorted(selected)[:350]
        previous = -2
        for i in indices:
            if i != previous + 1:
                sections.append("...\n")
            sections.append(f"{i + 1:5d}: {lines[i]}\n")
            previous = i
        if not indices:
            sections.append("NO MATCHING EXCERPTS\n")
    out.write_text("".join(sections), encoding="utf-8")


def run_tests(repo: Path, report: dict) -> None:
    import numpy as np

    paths = [repo / "rlt_online_rl/src", repo / "src", repo]
    sys.path[:0] = [str(p) for p in paths]
    module = importlib.import_module("rlt_online_rl.action_representation")
    config_module = importlib.import_module("rlt_online_rl.config")
    for name, imported in [("adapter", module), ("config", config_module)]:
        actual = Path(imported.__file__).resolve()
        report[f"{name}_module"] = str(actual)
        if repo not in actual.parents:
            raise RuntimeError(f"Imported {name} from outside requested repo: {actual}")

    cls = config_module.RLTOnlineRLConfig
    fixture = {
        "action_dim": 16, "proprio_dim": 16, "chunk_len": 50,
        "action_representation": "delta_chunk", "delta_action_dims": 14,
    }
    signature = inspect.signature(cls)
    supported = {k: v for k, v in fixture.items() if k in signature.parameters}
    omitted = sorted(set(fixture) - set(supported))
    report["synthetic_fixture_config"] = supported
    report["fixture_fields_unsupported_by_config_class"] = omitted
    if omitted:
        report["warnings"].append(
            f"Local config class does not declare {omitted}. No field was monkey-patched."
        )
    cfg = cls(**supported)
    q01 = -np.linspace(0.8, 1.3, 16, dtype=np.float32)
    q99 = np.linspace(0.9, 1.4, 16, dtype=np.float32)
    q01[14:], q99[14:] = 0.0, 1.0
    stats = module.QuantileStats(q01=q01, q99=q99)
    adapter = module.ActionRepresentationAdapter(rl_config=cfg, stats=stats)

    B, C, D = 4, 50, 16
    p = np.tile(np.linspace(0.10, 1.60, D, dtype=np.float32), (B, 1))
    p += np.arange(B, dtype=np.float32)[:, None] * 0.03
    p[:, 14:] = [0.13, 0.82]
    pn = p.copy()
    pn[:, :14] += np.linspace(0.04, 0.12, 14, dtype=np.float32)
    pn[:, 14:] = [0.24, 0.71]

    def make_action(state: np.ndarray, phase: float) -> np.ndarray:
        t = np.linspace(0, 1, C, dtype=np.float32)[None, :, None]
        j = np.arange(14, dtype=np.float32)[None, None, :]
        action = np.empty((B, C, D), dtype=np.float32)
        action[..., :14] = state[:, None, :14] + 0.025 * np.sin(3.0 * t + j + phase)
        action[..., 14] = (0.2 + 0.45 * t[..., 0])
        action[..., 15] = (0.8 - 0.40 * t[..., 0])
        return action

    a, an = make_action(p, 0.0), make_action(pn, 0.7)

    def expected_normalized(action: np.ndarray, state: np.ndarray) -> np.ndarray:
        # Independent specification: only indices [0,14) become joint deltas.
        represented = action.copy()
        represented[..., :14] -= state[:, None, :14]
        return (represented - q01) / (q99 - q01 + np.float32(1e-6)) * 2 - 1

    expected, expected_next = expected_normalized(a, p), expected_normalized(an, pn)
    raw_batch = {
        "proprio": p.copy(), "next_proprio": pn.copy(),
        "ref_chunk": a.copy(), "action_chunk": a.copy(),
        "next_ref_chunk": an.copy(),
    }
    prepared = adapter.prepare_training_batch(raw_batch)
    inferred = np.stack([adapter.normalize_ref_chunk(a[i].copy(), p[i].copy()) for i in range(B)])
    restored = np.stack([adapter.denormalize_to_abs_chunk(inferred[i].copy(), p[i].copy()) for i in range(B)])

    def check(name: str, actual: object, target: object) -> None:
        x, y = np.asarray(actual), np.asarray(target)
        result = {"name": name, "actual_shape": list(x.shape), "expected_shape": list(y.shape)}
        if x.shape != y.shape or not np.all(np.isfinite(x)):
            result.update(status="FAIL", reason="shape mismatch or non-finite output")
        else:
            error = np.abs(x.astype(np.float64) - y.astype(np.float64))
            maximum = float(error.max())
            result.update(
                status="PASS" if maximum <= 1e-5 else "FAIL",
                max_abs_error=maximum,
                per_dim_max_abs_error=error.max(axis=tuple(range(error.ndim - 1))).tolist(),
            )
        report["tests"].append(result)

    check("train_ref_semantics_14d", prepared["ref_chunk"], expected)
    check("train_action_semantics_14d", prepared["action_chunk"], expected)
    check("next_ref_uses_next_proprio", prepared["next_ref_chunk"], expected_next)
    check("train_vs_inference_normalization", prepared["ref_chunk"], inferred)
    check("reference_roundtrip_numpy", restored, a)
    check("gripper_semantics_absolute_before_normalization", inferred[..., 14:], expected[..., 14:])

    # Check the differentiable reverse path separately. It can differ from NumPy.
    fn = getattr(module, "jax_denormalize_to_abs_chunk", None)
    if fn is None:
        report["tests"].append({
            "name": "jax_inverse_semantics_14d", "status": "NOT_CHECKED",
            "reason": "Function missing/renamed; inspect local implementation.",
        })
    else:
        import jax.numpy as jnp
        options = {"action_representation": "delta_chunk"}
        if "delta_action_dims" in inspect.signature(fn).parameters:
            options["delta_action_dims"] = 14
        report["jax_inverse_signature"] = str(inspect.signature(fn))
        report["jax_inverse_unit_test_kwargs"] = options
        try:
            # Feed the independently correct representation, NOT the adapter output.
            result = fn(jnp.asarray(expected), jnp.asarray(p), jnp.asarray(q01), jnp.asarray(q99), **options)
            check("jax_inverse_semantics_14d", result, a)
        except Exception as exc:
            report["tests"].append({
                "name": "jax_inverse_semantics_14d", "status": "NOT_CHECKED",
                "reason": f"Local signature/implementation differs: {type(exc).__name__}: {exc}",
            })
    statuses = [r["status"] for r in report["tests"]]
    report["status"] = "FAIL" if "FAIL" in statuses else (
        "INCOMPLETE" if "NOT_CHECKED" in statuses or omitted else "UNIT_TESTS_PASS"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default="/tmp/rlt_step1_action_audit")
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        parser.error(f"Repository not found: {repo}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.output_root).expanduser() / timestamp
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "NOT_RUN", "repo": str(repo), "git_commit": git_rev(repo),
        "test_scope": "synthetic unit inputs + synthetic quantile statistics; NOT live runtime",
        "runtime_validated": False, "real_replay_validated": False,
        "real_normalization_stats_validated": False, "checkpoints_loaded": False,
        "synthetic_test_max_abs_tolerance": 1e-5, "tests": [], "warnings": [],
        "config_path": str(Path(args.config)),
    }
    try:
        config_path = Path(args.config).expanduser()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        report["recorded_config_entries"] = config_entries(payload)
    except Exception as exc:
        report["warnings"].append(f"Could not read recorded config: {type(exc).__name__}: {exc}")
    try:
        collect_sources(repo, out / "source_excerpt.txt")
        run_tests(repo, report)
    except Exception:
        report["status"] = "ERROR"
        report["traceback"] = traceback.format_exc()
    report_path = out / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSTATUS: {report['status']}")
    print(f"SCOPE: {report['test_scope']}")
    for test in report["tests"]:
        error = test.get("max_abs_error")
        suffix = f" max_abs_error={error:.8g}" if error is not None else ""
        print(f"{test['status']:>11}  {test['name']}{suffix}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    if report.get("traceback"):
        print(report["traceback"])
    print(f"\nSUMMARY: {report_path}\nSOURCES: {out / 'source_excerpt.txt'}")
    print("Stop here. This report does not certify replay semantics or environment commands.")
    return 0 if report["status"] == "UNIT_TESTS_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
PY_RLT_ACTION_AUDIT

# Record the exact script identity before execution.
"$PY" -B - "$WORK_DIR/rlt_step1_action_audit.py" <<'PY_AUDIT_HASH' | tee "$WORK_DIR/audit_script_sha256.txt"
from pathlib import Path
import hashlib
import sys
p = Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest(), p.name)
PY_AUDIT_HASH

# Preserve the Python exit code rather than the exit code from tee.
set +e
"$PY" -B -u "$WORK_DIR/rlt_step1_action_audit.py" \
  --repo "$REPO" \
  --config "$CONFIG" \
  --output-root "$WORK_DIR/reports" \
  2>&1 | tee "$WORK_DIR/execution.log"
PIPE_CODES=("${PIPESTATUS[@]}")
RUN_RC="${PIPE_CODES[0]}"
LOG_RC="${PIPE_CODES[1]}"
set -e
printf '%s\n' "$RUN_RC" > "$WORK_DIR/exit_code.txt"
printf '%s\n' "$LOG_RC" > "$WORK_DIR/log_exit_code.txt"

if command -v git >/dev/null 2>&1; then
  git -C "$REPO" status --porcelain=v1 --untracked-files=normal \
    > "$WORK_DIR/git_status_after.txt" 2> "$WORK_DIR/git_status_after.stderr" || true
  if [[ -f "$WORK_DIR/git_status_before.txt" ]]; then
    diff -u "$WORK_DIR/git_status_before.txt" "$WORK_DIR/git_status_after.txt" \
      > "$WORK_DIR/git_status.diff" || true
  fi
fi

printf '\nWORK_DIR: %s\nAUDIT_EXIT_CODE: %s\nLOG_EXIT_CODE: %s\n' \
  "$WORK_DIR" "$RUN_RC" "$LOG_RC"
printf 'Read SUMMARY and SOURCES paths in: %s/execution.log\n' "$WORK_DIR"
echo 'Now write the report requested in section 6, then stop. Do not repair or train.'
if [[ "$LOG_RC" -ne 0 ]]; then
  exit 2
fi
exit "$RUN_RC"

```

内嵌 Python 脚本的预期 SHA-256：

```text
a8e7fdbd898a5249dc7ee242a6be2ca65b4411e57f26571dd018b2786e3690df
```

以上脚本沿用上一轮提供的检查脚本；shell 包装新增了独立工作目录、CPU 环境、日志与退出码保存。若保存后的脚本哈希不符，先检查复制是否完整；不要修改测试逻辑。

## 6. 执行完毕后，Codex 必须整理的结果

从 `execution.log` 中读取 `SUMMARY:` 和 `SOURCES:` 后的**实际完整路径**，不要猜测时间戳目录。读取对应的 `summary.json`、`source_excerpt.txt`，并检查本次日志、退出码和 Git 状态差异。不要执行摘录中的训练或导出脚本。

在本次 `WORK_DIR` 下新建 `codex_report.md`，按以下结构填写：

```text
# RLT Step 1 审计结果

## 总体状态
- overall_status: UNIT_TESTS_PASS / FAIL / ERROR / INCOMPLETE
- 原始 summary.status:
- Python 退出码与日志保存退出码:
- 实际仓库路径、Git commit、adapter/config 模块路径:
- 本轮实际运行的命令及工作目录:

## 七项测试结果
填写每项名称、状态、最大误差；缺测写 NOT_CHECKED，不猜测。
失败时附逐维误差，指出是否集中在索引 6:14 或 14:16。

## 配置与源码证据
记录 action_dim、proprio_dim、chunk_len、delta_action_dims、action_representation。
记录实际读到的 gamma、action_norm_stats_path；缺失写“未找到”，不猜默认值。
检查 config 中的字段是否被本次测试配置类支持。
区分“配置文件声明了 14”“函数测试支持 14”“真实调用点传入了 14”。
如发现写死 :6、下一状态基准错误、JAX/NumPy 不一致或字段没有传递，
列出本地 文件:行号 及最小必要源码，并注明是数值验证还是静态线索。
若相关源码不在摘录范围，可继续只读查看相关函数与紧邻调用点。
不要把静态线索写成已经完成端到端验证。

## 未验证范围
没有验证真实 normalization stats、真实 replay、padding/有效 chunk 长度、
奖励和时间对齐、actor 行为、最终 env.step 命令或任务成功率。
没有加载 checkpoint，也没有证明 Stage-1/critic 已训练好。

## 文件变更与阻塞
列出本轮新建文件目录；说明是否出现仓库状态变化。
若有变化，区分执行前已有修改与本轮新发现的变化；不要恢复或删除。
若 Git 状态不可用，明确写“未验证”，不要宣称仓库完全无变化。

## 交付文件
列出 codex_report.md、summary.json、source_excerpt.txt、execution.log 的实际路径。
失败且未生成某个文件时，明确写“未生成”，附 blocked.txt 或 traceback。

## 停止声明
本轮已停止；未修复业务代码、未重训、未启动仿真、未进入下一步。
```

状态判定规则：

- 有数值不符合规格的测试：`FAIL`；有执行异常：`ERROR`。不要将函数接口不兼容当作数值错误，也不要把失败解释为已经找到了训练效果差的唯一原因。
- 全部测试通过但关键路径缺失、配置读取失败、字段不支持、JAX 路径未检查或报告保存不完整：`INCOMPLETE`。保留原始 `summary.json` 不动，在 `codex_report.md` 中解释差异。
- 七项都通过、必要配置可读取且规格一致、无未解决阻塞：`UNIT_TESTS_PASS`。这只表示合成单元测试通过。
- 即使配置或静态线索明显有问题，本轮也只报告，不修代码。

最后在对用户的回复中给出总体状态、一段有证据支持的结论，以及上述交付文件的实际路径。用户会将结果交回用于决定下一步；**不要自行执行第 2 步。**

## 7. 参考来源与证据优先级

证据优先级是：本地实测结果 → 本地源码与已落盘配置 → 用户提供的背景 → 公开仓库参考。文档中的动作规格来自用户描述；审计步骤、容差和停止条件是本次排查方案，不是论文规定的验收标准。

公开仓库的动作适配器提供了本检查使用的接口与 quantile 变换参考；其公开版本存在固定前 6 维的转换，因此本轮采用独立 14 维预期而非仅测试 round-trip。[S1] 这不代表用户本地版本也有同样问题。

[S1] `Yyshadow/openpi-RLT`，`rlt_online_rl/src/rlt_online_rl/action_representation.py`。本次编写时已通过网页重新核对；`main` 是可变分支，执行时仍以本地版本为准。

```text
https://raw.githubusercontent.com/Yyshadow/openpi-RLT/main/rlt_online_rl/src/rlt_online_rl/action_representation.py
```