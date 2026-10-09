# Codex 执行任务：RLT 第 2 步——真实 replay、动作统计量与奖励时间审计

> **直接执行本任务，生成报告后停止。** 本轮仍然只用 CPU，不加载模型权重，不训练，不启动仿真，不修改业务代码、配置、replay 或统计量。先根据本地源码确认真实文件格式和接口，再在独立临时目录编写、执行只读审计脚本。不要重新进行第 1 步的合成测试，也不要把公开仓库代码覆盖到本地。
>
> 本文包含完整任务规格、实际路径、CPU 环境初始化命令、独立数值参考和交付标准，不需要其他附件。**真实 replay 读取器和本地函数调用由 Codex 按本地实现编写**，不得臆测 pickle 布局或导入训练入口来自动恢复整个系统。

## 1. 已完成的检查与本轮问题

用户提供的第 1 步报告：七项 CPU 合成动作转换测试全部通过，最大绝对误差 `1.1920928955078125e-07`；测试前后 Git 状态一致。

已确认的规格与记录：

| 项目 | 值或结论 |
|---|---|
| 仓库 | `/root/workspace/rlt/RLT` |
| Python | `/root/workspace/envs/openpi/bin/python` |
| 上次 Git commit | `4271fedb817942bf48f69bc9109a315f96748393`；本轮重新记录，不强制回退 |
| 动作与状态 | 16 维；索引 `0:14` 为关节，`14:16` 为夹爪 |
| 表示 | 关节相对 chunk 起点 proprio 做 delta，夹爪保持 absolute，然后做 quantile normalization |
| chunk_len | 50；**这尚未证明每条 transition 实际执行了 50 个控制步** |
| gamma | 落盘 `rl_config.gamma = 0.99`；本轮核对它实际用于哪一条 TD 计算路径 |
| 特征版本 | replay 使用 Stage-1 **20k** checkpoint；不切换到 200k |
| Stage-1 | VLA 冻结、`rlt_alpha=0`；本轮不评估 token 质量 |
| replay 预期 | 4,122 条 transition，150 个 episode，50 成功、100 失败；全部 BASE/warmup，`action_chunk == ref_chunk` |
| 序列化动作 | journal 中应为 absolute；送网络前转换为 delta 并归一化 |
| Stage-2 权重 | 离线 BC/Q/delta = `10 / 0.001 / 100`，Q 从第 5,000 步加入；本轮不修改 |
| Step-1 已验证范围 | 合成统计量下的训练/推理动作转换、14 维语义、next_proprio 基准与 NumPy/JAX 逆转换 |

**本轮只回答三件事：**

1. 真实 replay 经真实 adapter 和真实 norm stats 做零修正透传，能否正确还原已保存动作？统计量与动作表示是否匹配？
2. 每条 transition 的 reference、实际动作、奖励、下一状态是否对应正确的时间区间？有效执行长度和 padding 是否能确定？
3. `gamma=0.99` 在实际控制时间尺度上产生多大的奖励信号？本地 TD target 的奖励与终止 mask 计算是否符合其定义？

不要把本轮结论扩展为“critic 已经训练好”“Stage-1 已经训练好”或“仿真执行已验证”。

## 2. 输入路径

```text
CONFIG=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2_retrain_conservative_8gpu/analysis/stage2_training_config.json
REPLAY=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2/replay_source/replay/replay_journal.pkl
MANIFEST=/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2/replay_source/replay_manifest.json
EXPECTED_STATS=/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05/assets/norm_stats.json
ROLLOUT_ROOT=/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks
```

真实统计量路径以本轮实际读取的 CONFIG 为准。若与 EXPECTED_STATS 不一致，分别记录两者、文件身份和来源，不静默替换。MANIFEST 是用户之前给出的路径；缺失时记录，允许在 `replay_source` 顶层及直接子目录只读查找被 exporter 明确命名的 manifest，不允许递归扫描整台机器。

原始 rollout 用于少量时间证据抽查。先读 exporter/manifest 确认格式和 episode 映射，不猜文件名，不因为 `episode_id=100` 就假设是目录排序第 101 项。**禁止解码图像、视频或加载完整视觉缓存。**

## 3. 安全与资源边界

所有新文件放在 `/tmp/rlt_step2_replay_audit/` 的本次独立目录，以下简称 `WORK_DIR`。业务仓库只读；保留用户已有修改和未跟踪文件。不安装依赖，不联网，不改全局环境，不操作其他训练进程。

本轮不需要 GPU。对本次所有 Python 子进程设置 `CUDA_VISIBLE_DEVICES=""`、`JAX_PLATFORMS=cpu`，并在需要导入 JAX 时记录设备列表，要求全部为 CPU。[S4]

仅允许轻量配置类、动作 adapter、纯函数与 NumPy/JAX 数值运算。**不得实例化 ReplayManager、按训练配置 capacity 分配 ReplayBuffer，或构造/初始化实际 actor、critic、VLA。** 不运行 `train_*.py`、`export_*.py`、`compute_norm_stats.py`，也不执行这些文件的 `main()`。

资源与读取规则：

- 开始时记录 journal 字节数、系统可用内存及可读的容器内存限制。优先流式读 journal，小批处理，最多保留少量完整 transition 样例；不要把全体 token/动作重复拼成多个大数组。
- 本任务设置审计预算：journal 大于 **1 GiB**，或格式要求一次物化超过 **256 MiB** 的单个对象且不能安全分块时，记录 `INCOMPLETE_RESOURCE_GUARD`，停止该数据分支。不得为完成审计而申请 GPU 或释放其他进程内存。这是本轮资源预算，不是对正常 replay 格式的限制。
- 分位数诊断可使用固定 seed=42、最多 100,000 行的动作 reservoir；全量计数和误差最大值仍用流式累计。记录抽样方法和覆盖量。token 只检查形状、dtype、有限值和必要的端点一致性，不保存全体 embedding。
- 对原始 rollout 只读少量数值/元数据切片。大型 pickle 若包含全部图像，不能为了拿几项元数据而完整反序列化；改查已有轻量日志，否则标记证据不足。
- pickle 只允许读取上述用户自己生成、可信的本地 journal/关联数值记录。Python pickle 反序列化不是安全沙箱；不得读取来历不明的下载文件。[S5]
- 读取前后记录 journal 的大小与 `mtime_ns`。如正在变化，不分析为静态完整数据，不停掉写入者，报告 `INCOMPLETE_INPUT_CHANGED`。
- 不将 `/tmp` 中已有目录、已有报告或业务文件删除。依赖缺失、接口不兼容、来源不明时保留错误，不通过 monkey-patch 或默认值偷偷补齐。

允许修改**本轮新建的审计脚本**来适配经源码确认的容器格式、函数签名或修正脚本自身错误；保留执行记录。禁止改变验收期望、修改被测函数，或为得到 PASS 扩大容差。

## 4. 初始化 CPU 工作目录

用 Bash 执行：

```bash
set -euo pipefail
BASE="/tmp/rlt_step2_replay_audit"
mkdir -p "$BASE"
WORK_DIR="$(mktemp -d "$BASE/codex_XXXXXXXX")"
REPO="/root/workspace/rlt/RLT"
PY="/root/workspace/envs/openpi/bin/python"
AUDIT_CONFIG="/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2_retrain_conservative_8gpu/analysis/stage2_training_config.json"
AUDIT_REPLAY="/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2/replay_source/replay/replay_journal.pkl"
AUDIT_MANIFEST="/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/stage2/replay_source/replay_manifest.json"
AUDIT_EXPECTED_STATS="/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05/assets/norm_stats.json"
AUDIT_ROLLOUT_ROOT="/mnt/pfs/kk/kk/data/data/geniesim/rollout/stack_three_blocks"

mkdir -p "$WORK_DIR/cache" "$WORK_DIR/tmp" "$WORK_DIR/reports"
printf '%s\n' "$WORK_DIR" > "$WORK_DIR/work_dir.txt"
{
  for name in WORK_DIR REPO PY AUDIT_CONFIG AUDIT_REPLAY AUDIT_MANIFEST AUDIT_EXPECTED_STATS AUDIT_ROLLOUT_ROOT; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
  cat <<'ENV_CPU'
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$WORK_DIR/cache"
export TMPDIR="$WORK_DIR/tmp"
export PYTHONPATH="$REPO/rlt_online_rl/src:$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
ENV_CPU
} > "$WORK_DIR/env.sh"
source "$WORK_DIR/env.sh"
printf 'WORK_DIR=%s\nENV_FILE=%s/env.sh\n' "$WORK_DIR" "$WORK_DIR"

if [[ ! -d "$REPO" || ! -x "$PY" || ! -r "$AUDIT_CONFIG" || ! -r "$AUDIT_REPLAY" ]]; then
  printf 'INCOMPLETE: required repo/python/config/replay path is unavailable.\n' | tee "$WORK_DIR/blocked.txt"
  exit 2
fi

git -C "$REPO" rev-parse HEAD > "$WORK_DIR/git_commit.txt" 2> "$WORK_DIR/git_commit.stderr" || true
git -C "$REPO" status --porcelain=v1 --untracked-files=normal \
  > "$WORK_DIR/git_status_before.txt" 2> "$WORK_DIR/git_status_before.stderr" || true
stat -- "$AUDIT_CONFIG" "$AUDIT_REPLAY" > "$WORK_DIR/input_stat_before.txt"
cat /proc/meminfo > "$WORK_DIR/meminfo_before.txt"
for name in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  if [[ -r "$name" ]]; then
    printf '\n%s\n' "$name" >> "$WORK_DIR/cgroup_memory_before.txt"
    cat "$name" >> "$WORK_DIR/cgroup_memory_before.txt"
  fi
done
printf 'Initialization complete. Inspect local sources before implementing the reader.\n'
```

Codex 记住命令实际打印的 WORK_DIR。后续若是新的 shell，先 `source` **本次实际目录**中的 `env.sh`，不要把示例目录或上一轮目录当成当前路径。

## 5. 先确认本地数据链路，再实现读取器

优先只读这些本地文件的相关函数与紧邻调用点，引用准确 `文件:行号`：

```text
scripts/geniesim/export_rollout_replay.py
scripts/geniesim/train_stage2_plan.py
rlt_online_rl/src/rlt_online_rl/replay.py
rlt_online_rl/src/rlt_online_rl/action_representation.py
rlt_online_rl/src/rlt_online_rl/config.py
rlt_online_rl/src/rlt_online_rl/networks.py
rlt_online_rl/src/rlt_online_rl/trainer.py
rlt_online_rl/src/rlt_online_rl/inference.py
```

根据实际 import 和配置引用，可继续只读其直接相关的 GenieSim 动作映射/采集代码、SFT 数据 transform 和统计量生成配置。不要执行训练配置工厂中可能触发数据集/模型下载的逻辑；优先静态读取。

必须先回答：

- journal 是连续 `pickle.dump(record)`、一个 list、一个 dict 还是其他布局？是否有 header、批记录或多个版本？公开实现是追加 pickle 记录，但以本地 exporter 和 writer 为准。[S1]
- `step_id` 是 chunk 序号、环境控制步号还是其他索引？chunk stride 是多少？是否有重叠、末尾对齐窗口、action repeat 或控制降采样？
- reference 是观测时保存的原始 VLA proposal，还是从后续执行动作拼接、或离线用模型重新生成？`next_ref_chunk` 从哪里来？
- journal 的 action/proprio 维度顺序、单位、dtype 和 padding 约定是什么？从原始控制命令到 16 维字段是否有重排？
- 本地训练实际使用哪个 TD target 函数？chunk 有效长度是否进入奖励折扣和 bootstrap？`done` 如何由成功、失败、超时形成？

在 WORK_DIR 下编写 `audit_step2.py`（也可以拆成少量明确命名的模块）。流式 journal 必须读到干净的对象边界 EOF；**已经开始读取一条记录后发生的 EOF/UnpicklingError 不能当成正常结束**。不调用带写入/分配副作用的 ReplayManager 来简化读取。

读取 CONFIG 中真实的 `rl_config` 构造配置/adapter。记录字段名、实际导入路径、stats loader 返回的维度和文件身份。不得通过“只抄 action_dim 等四个字段”又重建一个合成配置来冒充真实配置。无关字段若无法构造，可依据本地解析器的同等纯逻辑解析，但必须记录差异；关键字段未知即标记 INCOMPLETE。

## 6. 必做 A：真实 replay 全量结构清点

对可安全读取的全量 journal 累计：

- transition 总数、unique episode 数、`source`/`source_chunk`/collection_phase 分布、`action_chunk == ref_chunk` 的数量和最大偏差。
- 必需字段、shape、dtype、非有限值数量；分别验证 z/next_z 长 2048、proprio/next_proprio 长 16、三个动作块为 `50×16`、rewards 长 50。其它布局只能据本地定义解释，不能自动 reshape 消除问题。
- 按 episode 记录 transition 数、step_id 序列特征、done 数、非零奖励位置、奖励值、success 标签分布及 episode 最终结果来源。
- 查重 `(episode_id, step_id)` 并展示样例；区分真正重复、冲突记录和有说明的不同窗口类型，不自动去重。
- 检查是否存在跨 episode 拼接、结束后的数据、缺失端点等迹象。**重叠窗口可能使同一终点出现在多条 done transition 中，不能预先要求每个 episode 只能有一个 done。**
- 分开报告 `success` 是末尾标签还是沿 episode 广播标签。不能要求成功 episode 的每条 transition 都有 `success=1`。
- 如果窗口重叠，不把 reward_chunk 中重复出现的同一个环境奖励相加当成 episode 总奖励。

与用户预期 4,122 / 150 / 50 / 100 对照。差异须解释或标记 INCOMPLETE，不以用户数字覆盖实测结果。

输出 `replay_inventory.json` 与 `episode_audit.csv`。CSV 至少包含：episode_id、transition_count、final_outcome、outcome_evidence、done_window_count、reward_window_count、control_steps_verified、length_evidence、overlap_or_stride、notes。缺证据写空值/UNKNOWN，不填推测值。

## 7. 必做 B：真实统计量与零修正透传

### B1. 统计量本身与来源

读取真实 adapter 的 stats，不修改：

- 记录实际文件路径、大小、mtime、SHA-256、JSON 键路径、原始 q01/q99 长度，以及 loader 实际选用的 16 维向量。若来自 32 维，必须记录本地如何选取，不能擅自切前 16 维救活测试。
- 检查 q01/q99 有限性、q99 小于 q01 的异常，以及近零跨度维度。q01=q99 可能来自常量维度，先结合真实变化范围诊断，不能仅因此判定文件错误。
- 追踪 stats 生成时统计的是 absolute、chunk 起点 delta、相邻步 delta，还是其他表示；检查动作维度顺序和 gripper 位置与 replay 是否一致。
- **stats 放在 SFT checkpoint 目录，不足以证明它是 absolute stats。** OpenPI 的统计流程可先执行 data transforms 后再统计，所以要看本地生成配置与保留的日志/元数据。[S3]
- 有当前源码只能证明当前生成逻辑；没有历史命令/配置/文件身份就不能声称已经证明历史 stats 来源。记录 `VERIFIED_METADATA / STATIC_COMPATIBLE / UNKNOWN` 证据等级。

### B2. 数值检查与 padding 分组

不加载 actor。让送入 adapter 的 reference 原样作为“actor 均值”，完成：

```text
saved absolute reference
→ 使用真实 adapter / stats 转 delta 并归一化
→ 不作任何修正
→ 使用真实 adapter 反归一化、加回 proprio
→ 与 saved absolute reference 比较
```

对实际 action 做同样检查；next_ref 必须用 next_proprio。还要比较训练批量入口与逐条推理入口，以及独立 14 维公式与本地转换，不能仅验证互逆。

对已确认有效的动作行，使用下方独立参考。比较基准是**journal 已保存的数值转 float32**；若 journal 原本是 float16，不把此前序列化量化误差算成此次 adapter 误差。

```python
# 独立参考：保存为 WORK_DIR 下的 reference_math.py，不导入业务 adapter。
import numpy as np


def reference_normalize(absolute, state, q01, q99):
    a = np.asarray(absolute, dtype=np.float32).copy()
    p = np.asarray(state, dtype=np.float32)
    lo = np.asarray(q01, dtype=np.float32)
    hi = np.asarray(q99, dtype=np.float32)
    if a.shape[-1] != 16 or p.shape[-1] != 16 or lo.shape != (16,) or hi.shape != (16,):
        raise ValueError("Expected 16D actions/state and resolved 16D statistics")
    while p.ndim < a.ndim:
        p = np.expand_dims(p, axis=-2)
    a[..., :14] -= p[..., :14]
    return (a - lo) / (hi - lo + np.float32(1e-6)) * np.float32(2) - np.float32(1)


def discounted_reward(rewards, gamma):
    r = np.asarray(rewards, dtype=np.float64)
    return np.sum(r * np.power(float(gamma), np.arange(r.shape[-1])), axis=-1)


def reference_td(rewards, done, gamma, min_next_q, elapsed_steps):
    # rewards 应与真实步索引对应；有效长度不明时不能拿此函数猜测真实语义。
    r = discounted_reward(rewards, gamma)
    d = np.asarray(done, dtype=bool)
    b = np.power(float(gamma), np.asarray(elapsed_steps)) * np.asarray(min_next_q)
    return r + np.where(d, 0.0, b)
```

审计脚本需先用小输入测试自身：reference_normalize 前 14 维减 state、夹爪不减；`gamma=1` 时 discounted_reward 等于求和；done=True 时 reference_td 不受有限 next_q 改变影响。这是脚本自检，不代替本地函数测试。

padding 处理必须分开：

- 有明确 valid_length/action_mask 或可核对原始步数时，按证据分组有效行与补齐行，记录补齐值和 mask 传播。
- 全零行或近零行只标记为 `ZERO_ROW_CANDIDATE`，**不能仅凭数值为零认定为 padding 并排除**。有效的零动作被 adapter 当作 padding，是另一类需报告的问题。[S2]
- terminal 的 next_ref 为占位值可以是合法设计，不能把它一概视为动作错误；检查实际训练如何屏蔽。非有限占位值也要报告，不能假设乘以零就安全。
- 对已知 padding，观察转换结果、是否进入 BC/delta/critic 输入或损失；不要要求 padding round-trip 与真实动作完全相同，也不要把没有排查的 padding 写为通过。

误差采用预先声明的数值规则 `abs(error) <= 1e-5 + 1e-6 * abs(reference)`，记录绝对误差、相对尺度、逐维误差和超过阈值的数量。该规则针对此次运算舍入，不是机器人精度标准。超过时保留数值，区分算术实现差异、条件数问题和语义错误，不临时放宽规则。

完整 NumPy 检查小批流式覆盖全量可判定有效数据。JAX 逆变换只抽最多 32 条代表性 transition 检查（成功/失败、首/中/末、发现的异常），CPU 上小批运行，并记录覆盖范围。

### B3. 动作尺度诊断（只报告，不改 stats）

对 ref/action/next_ref 分别汇总每一维：真实表示的 min/max/分位数、归一化后分位数、`abs(normalized)>1/2/5/10` 比例、near-zero stats span，以及物理单位是否有证据。

**超出 [-1,1] 不自动代表归一化错误**：quantile 映射不是强制裁剪，也不保证新分布的所有数据都落在分位区间内。[S2] 不自动 clip，不重算/覆盖正式统计量。统计量语义错误必须有生成链路或真实数据证据，不能单凭某个阈值下结论。

输出 `normalization_audit.csv` 和 summary 中的 real_roundtrip_tests。失败样例附 episode/step/行/维度、原始数值、stats、期望和实测数值。

## 8. 必做 C：真实时间区间与 reference 来源

先以元数据尽可能确认全量 episode 的长度/stride；然后确定性抽查**2 个成功、2 个失败 episode**。优先包含已知成功 episode 100 和失败 episode 0；若不存在，用实测结果替代并记录，不能强行映射。每个 episode 检查首段、中段、末段；优先加入短末段或重叠边界，共不超过 16 个窗口。

每个窗口需记录下列证据，缺失写 UNKNOWN：

```text
episode_id, replay_step_id
observation 的原始索引/时间戳
VLA proposal 的原始生成索引/标识及预测 horizon
实际执行动作的 [start, stop) 控制步范围
该 chunk 实际执行的控制命令数，是否 action_repeat/降采样
rewards 对应的环境步范围和成功/终止发生的步号
next observation 的原始索引/时间戳
valid_length、padding、下一窗口起点
数据来源文件及字段 / 代码行号
```

重点判断：

1. 是“预测 50、实际执行 50”，还是“预测 50、实际只执行 K 就重规划”？不能用 shape 推断 K。
2. `ref_chunk == action_chunk` 的原因，是完整执行了同一个起点 proposal，还是 exporter 用未来执行序列复制了 reference？若后者包含后续重规划结果，要明确标记其非因果 reference 风险；若没有保存 proposal，不能假装已经恢复。
3. 下一状态是否为执行完该窗口后的观测，而不是一帧后、提前/滞后一段，或下个 episode 的首帧？
4. 最后一次真实奖励是否放在 `valid_length-1` 的正确位置？如果 exporter 把不足 50 步的末段奖励统一移到索引 49，直接记录证据，但不修改。
5. 对 contiguous、非重叠且端点明确匹配的窗口，才比较 current.next_proprio 与对应 next.proprio。z/next_z 仅在相同 token 版本和提取约定下比较，记录 dtype/量化差异；不能按文件相邻记录盲比。
6. terminal/truncation 的含义是否对应任务定义？有限时长任务的超时可能是任务终止，外部采集截断则未必；不能把所有 timeout 自动改成需要 bootstrap。[S6]

输出 `time_alignment_samples.csv`。对于只靠当前代码推导、没有原始日志支持的结论，明确 `STATIC_ONLY`；日志实证才写 `VERIFIED_SAMPLE`。本轮抽样通过不等于全量轨迹时间对齐都已实证。

## 9. 必做 D：TD target 单元测试与折扣量级

### D1. 本地纯函数测试，不初始化网络

定位本地训练**实际调用**的 target builder。[S7] 先静态确认其参数/调用链，再调用原函数进行 CPU 测试。

允许在审计脚本定义轻量 actor/critic **测试替身**：actor 返回传入 next_ref，critic 返回预设的常量 Q。将这些对象作为函数参数传入；**禁止 monkey-patch 被测模块、加载 checkpoint、构造实际模型参数或启动 learner**。这是测试 Bellman target 算法，不是测试学到的 Q。

若本地函数封装要求真实模型/编译完整 learner，停止该子测试，报告 `LOCAL_TARGET_NOT_EXECUTED`，保留静态公式证据。不能用自己重写的公式得到一致结果后声称测试了本地实现。

对固定完整 C=50、gamma=0.99 的基本定义：

```text
y = sum(gamma**i * r[i], i=0..49)
    + (1-done) * gamma**50 * min(next_q)
```

最少检查：

| 用例 | 输入 | 独立预期 |
|---|---|---|
| 成功终止 | r[49]=1，其余 0，done=True | gamma**49 |
| 失败终止 | r 全 0，done=True | 0 |
| 终止 mask | 上面两例分别用有限 next_q=-3、0、7 | target 不变 |
| 非终止 bootstrap | r 全 0，done=False，min(next_q)=2 | 2*gamma**50 |
| 奖励索引 | r[0]=1，done=False，min(next_q)=2 | 1+2*gamma**50 |
| 双 Q 选择 | next_q 分别 5 和 2，交换顺序 | 若本地定义用 min，两次都选 2 |

本地若用不同但明确配置的 target 定义，记录实际定义和来源，不能悄悄将其改成上式来通过测试。数值比较容差同 B2；将独立数学预期与 LOCAL_FUNCTION 输出分列。

随后对真实成功/失败 terminal 奖励向量做同类检查，不强制所有成功 reward 都在索引 49。若真有短有效窗口，分别列：**当前实现算出的 target**、**按已验证实际奖励步索引算出的 target**、二者差值和证据。有效长度/时间不明则不能计算“真实 target”。

### D2. 量化 gamma，不修改 gamma

先记录 `0.99**49`、`0.99**50`。然后按**已验证的实际控制步长**报告成功轨迹起点的经验折扣回报及其 min/median/max。只有确认“每个环境步奖励、唯一末端奖励 1”时，长度 T 的起点回报才写成 `gamma**(T-1)`。

可以附上 gamma=0.999、0.9995 的同轨迹敏感性计算，标注“假设比较，不是实际训练配置，也不是本轮改参建议”。数据有中间奖励时，对完整真实奖励序列计算，不套唯一终点公式。

**禁止用 transition_count*50 或 (max(step_id)+1)*50 直接推断 T，除非非重叠执行、完整长度、索引语义都已经核实。重叠 replay 不能通过拼接 reward_chunk 来构造 MC return。**

若没有可验证的环境步数，仅报告 chunk 内的折扣和“时间尺度未确认”，不要复用此前聊天中的 1,500 步假设充当本次实测。

较小的合法回报归类为 `DISCOUNT_SIGNAL_WARNING`，不是代码 FAIL；gamma 与 elapsed_steps 的实际使用不匹配，才是可定位的实现/数据问题。

## 10. 执行、保存日志与状态

审计脚本入口约定为：

```text
$WORK_DIR/audit_step2.py
  --config $AUDIT_CONFIG
  --replay $AUDIT_REPLAY
  --manifest $AUDIT_MANIFEST
  --rollout-root $AUDIT_ROLLOUT_ROOT
  --repo $REPO
  --output-dir $WORK_DIR/reports
```

脚本执行时每个已完成检查立即落盘结果；错误写入 summary/日志，保留已完成部分。子项接口/证据不足可以继续不依赖它的检查；缺少关键输入、检测到输入变化或资源风险则停止相关读取。数值错误不得掩盖为普通警告。

确认本次 `env.sh` 已 source 后执行：

```bash
set -euo pipefail
"$PY" -B - "$WORK_DIR/audit_step2.py" <<'PY_SYNTAX'
from pathlib import Path
import ast, hashlib, sys
p = Path(sys.argv[1])
b = p.read_bytes()
ast.parse(b.decode("utf-8"), filename=str(p))
print("syntax_ok", p)
print("sha256", hashlib.sha256(b).hexdigest())
PY_SYNTAX

set +e
"$PY" -B -u "$WORK_DIR/audit_step2.py" \
  --config "$AUDIT_CONFIG" --replay "$AUDIT_REPLAY" \
  --manifest "$AUDIT_MANIFEST" --rollout-root "$AUDIT_ROLLOUT_ROOT" \
  --repo "$REPO" --output-dir "$WORK_DIR/reports" \
  2>&1 | tee "$WORK_DIR/execution.log"
CODES=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${CODES[0]}" > "$WORK_DIR/exit_code.txt"
printf '%s\n' "${CODES[1]}" > "$WORK_DIR/log_exit_code.txt"

git -C "$REPO" status --porcelain=v1 --untracked-files=normal \
  > "$WORK_DIR/git_status_after.txt" 2> "$WORK_DIR/git_status_after.stderr" || true
diff -u "$WORK_DIR/git_status_before.txt" "$WORK_DIR/git_status_after.txt" \
  > "$WORK_DIR/git_status.diff" || true
stat -- "$AUDIT_CONFIG" "$AUDIT_REPLAY" > "$WORK_DIR/input_stat_after.txt"
printf 'WORK_DIR=%s\nAUDIT_EXIT_CODE=%s\nLOG_EXIT_CODE=%s\n' \
  "$WORK_DIR" "${CODES[0]}" "${CODES[1]}"
```

把审计脚本及 reference_math.py 的 SHA-256 记录在报告。核对配置与 stats 文件执行前后身份；输入异常不恢复、不修复。Git 状态不变只说明状态快照一致，不等同于对所有文件做了字节级校验。

## 11. 交付与状态判定

本轮输出保存在 WORK_DIR，至少包含：

```text
codex_report.md                    给用户复制的主报告
reports/summary.json              各检查结论、覆盖率、证据等级、阻塞
reports/replay_inventory.json     全量结构计数与字段定义
reports/episode_audit.csv         每个 episode 的长度/终止/奖励信息
reports/normalization_audit.csv   动作逐维尺度与透传误差
reports/time_alignment_samples.csv  少量原始日志时间对齐证据
source_excerpt.txt               精确文件:行号与最小相关源码
execution.log                    原始运行日志
```

不可完成的文件注明未生成及原因，不伪造空的 PASS 文件。不要把完整 replay、权重、图像或全量 embedding 打包进交付物。

`summary.json` 顶层至少包含：

```text
overall_status: PASS / PASS_WITH_WARNINGS / FAIL / INCOMPLETE / ERROR
scope: cpu_real_replay_audit_no_model_no_sim
input_identity, effective_config, imported_modules, devices
replay_inventory, normalization_tests, normalization_provenance
timing_verification, td_target_tests, discount_analysis
warnings, blockers, unverified, artifact_paths
```

每个 check 带 `status`、`evidence_level`（NUMERICAL / RAW_LOG_SAMPLE / STATIC_ONLY / UNKNOWN）、覆盖数量、误差/指标与对应文件行号或 episode/step。目标函数自写参考只能标 `INDEPENDENT_REFERENCE`，不能写 LOCAL_FUNCTION_TEST。

总体判定：

- **FAIL**：已证实的动作表示/透传错误、奖励/next-state 对齐错误、TD 数值或终止 mask 错误、非有限实际训练输入等。附最小复现，不自行修复。
- **INCOMPLETE**：关键 stats 来源/时间长度/reference 来源/本地 target 调用证据缺失，或资源/输入一致性限制；已完成的数值 PASS 保留。不能因为没有发现错误就将 UNKNOWN 改成 PASS。
- **PASS_WITH_WARNINGS**：所要求的检查在明确覆盖范围内完成，无确认的硬错误，但存在合法的较小折扣信号、显著分布变化等诊断警告。
- **PASS**：所要求的检查在明确覆盖范围内完成，无硬错误和未解决关键证据缺口。它只覆盖本轮数值与时间抽查，不是端到端训练验收。
- **ERROR**：审计程序/环境异常阻止必要检查。若局部已发现硬错误，也必须列出，不因总体 ERROR 丢失。

主报告 `codex_report.md` 开头用不超过一页回答：

1. 真实 stats 下零修正透传是否通过？有没有 padding/维度顺序/统计量表示问题？
2. gamma 实际值与实际执行长度是否已核实？是否有“预测 50 但只执行 K”的情况？
3. ref_chunk 与 action_chunk 相等的来源是否可验证为因果正确？
4. 真实成功末段的 reward 索引和 TD target 是多少？本地函数是否实际测过？
5. 是否有足够证据说折扣信号很小？哪些结论仍是未知？

后面附关键表格、原始样例证据、未验证范围、新建文件与 Git 状态、所有交付文件实际路径。主报告中的结论只用数据/源码支撑，不根据主观“看起来正常”下判断。

**完成后停止，不进入下一步，不改 gamma，不改 BC/Q/delta，不切 token checkpoint，不导出新 replay，不重算正式 norm stats。** 用户会先将 `codex_report.md` 和 `summary.json` 反馈，再决定下一轮。

## 12. 参考与证据优先级

证据优先级：本地实测与原始记录 → 本地实际代码/落盘配置 → 用户已提供报告 → 公开代码/官方文档。下列公开资料仅用于解释为什么检查这些环节，不证明本地修改版有同样行为。主分支可变；无需 Codex 联网重新获取。

- [S1] 公开 RLT replay 实现包含追加 journal 与预分配 buffer 路径；读取本地 journal 前检查实际布局，避免通过管理器恢复数据带来额外分配/写入。
  `https://raw.githubusercontent.com/Yyshadow/openpi-RLT/main/rlt_online_rl/src/rlt_online_rl/replay.py`
- [S2] 公开动作 adapter 的 quantile 公式、zero-row 特殊处理与反变换接口；本地 14 维修改已通过 Step-1，不能回退公开版。
  `https://raw.githubusercontent.com/Yyshadow/openpi-RLT/main/rlt_online_rl/src/rlt_online_rl/action_representation.py`
- [S3] OpenPI 官方 stats 计算流程先应用 repack/data transforms，解释了为什么不能仅凭 SFT 文件位置推断统计量表示。
  `https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/scripts/compute_norm_stats.py`
- [S4] JAX 官方平台配置说明：JAX_PLATFORMS 控制初始化的平台列表。
  `https://docs.jax.dev/en/latest/config_options.html`
- [S5] Python 官方 pickle 安全说明：只反序列化可信数据。
  `https://docs.python.org/3/library/pickle.html`
- [S6] Gymnasium 官方关于 termination/truncation 与 bootstrap 的说明；实际任务语义仍需本地证据。
  `https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/`
- [S7] 公开 RLT 的 build_td_target 和 discounted chunk rewards；本轮必须核对实际训练所用的本地目标函数。
  `https://raw.githubusercontent.com/Yyshadow/openpi-RLT/main/rlt_online_rl/src/rlt_online_rl/networks.py`

本任务中的检查顺序、抽样规模、数值容差、资源预算和停止条件均为本次排查设计，不是论文的官方验收标准。