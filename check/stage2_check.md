# RL Token 质量第二轮验证：Codex 执行文档

> **本文件中的“第二轮”指 Stage 1 RL token 的质量验证，不是 RL Token 论文中训练 Actor/Critic 的 Stage 2。** 阅读本文后在用户本地 GenieSim/openpi-RLT 工程中完成必要代码适配，实际运行离线实验，交付结果、图和中文报告。不要训练或部署 Actor/Critic，不进行 Actor 在线 rollout，也不要把本任务改写成 RL 训练方案。

## 0. 本轮要回答的三个问题

1. Stage 1 decoder 重建变好时，是否真的使用了每个观测对应的 `z_rl`？如果把 token 换成别的观测的 token，性能会怎样？
2. 在堆叠对齐、释放、回撤**发生错误之前**，`z_rl` 是否保留当前几何状态以及下一段执行风险的信息？
3. 在已有 `robot state + ref_chunk` 的条件下，再加 `z_rl` 是否提供可重复的增量信息？10k 与 170k 之间是否有可信的差别？

用“观测条件化的 token”作评估对象；不能用固定可学习 token 参数、decoder 输出、未来图像或动作结果冒充当时的 `z_rl`。本轮只训练用于诊断的轻量线性探针；若做 no-token 重建对照，可训练独立 decoder。它们不是 RL Actor、Critic，也不参与部署。报告中的分类 AUC、二维散点、低 loss 都不是预设的通过阈值。

## 1. 已知输入和执行范围

| 项目 | 用户当前信息；执行时核对真实文件 |
| --- | --- |
| 任务 | GenieSim 3 三积木堆叠；重点为中途堆叠高度/XY 对齐，以及放块后回撤碰撞 |
| Stage 1 checkpoint 根目录 | `/mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/checkpoints/rlt_pi05_geniesim_stack_three_blocks_plan/stage1_rl_token_20k/` |
| 基础 π0.5 | `/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05` |
| 原始 rollout | `/mnt/pfs/kk/kk/data/data/geniesim/rollout` |
| 主要对照 | 10k 和 170k；按 checkpoint 内实际 global_step 核实，不能凭根目录 `20k` 猜测 |
| 动作/观测时序 | π0.5 预测 H=50，原有执行窗口为 50 ticks，30 Hz，delta joint 相对 chunk 起点；每次新 policy-call 间隔通常约 1.67 秒，以真实日志核实 |
| 既有报告 | 20 成功 + 20 失败、每条取 16 calls，共 640 个点；170k NMSE=0.3269、cosine=0.9563、成败线性探针 AUC=0.7510；10 条不足 16 calls 的早停失败未进入该报告 |

先读适用的 `AGENTS.md`、检查 `git status`，保留现有改动和原始数据。定位 Stage 1 的 `src/openpi/models/rl_token.py`、`src/openpi/models/pi0.py`、`scripts/train_rlt.py`、`scripts/serve_rlt_policy.py`，以及本地的数据加载、重建指标计算和上次报告脚本。以**当前运行机器上的实现**为准，公开仓库只是结构线索；记录代码 commit、权重、预处理、相机、任务文本、归一化与 token 提取位置。

本轮可以从已有原始 rollout 读取图像、机器人状态、π0.5 参考动作和仿真几何真值；若需补采，仅允许在 GenieSim 中采集 **π0.5 BASE** 的观测/标签，且要与现有数据来源分开标记。不调整 50 步执行窗口、不启动真实机器人或 RL Actor。

## 2. 共同数据与严格的时间定义

### 2.1 清点样本，而不是继续只看 640 个点

- 列出原始数据中成功/失败/早停/timeout/未知 episode 的实际数量、`instance_id`、初始场景 seed、Stage 1 训练数据是否与它们重叠、图像时间戳与保存频率。上次 40 条可视化样本可复用，但关键事件分析应尽量使用全部可核实的 episode，尤其是此前排除的 10 条短早停失败。
- 保存 episode 清单 `episodes.csv` 和逐次 policy-call 清单 `calls.csv`；区分仿真 30 Hz **控制 tick**、实际保存图像时刻和模型请求时刻。不得把相距约 50 ticks 的两个 policy call 当成连续 30 Hz 图像；不得用事件之后的画面冒充事件之前的输入。
- 每个观测记录唯一 `sample_id`、episode/instance/seed、最终成败标签、真实调用时间/控制 tick、堆叠层数或方块序号、机器人状态、同步三路图像、当前调用已知的 `ref_chunk`、原始阶段信息和缺失原因。未来实际执行结果/最终成败只作分析目标，绝不进入 encoder 或探针的输入特征。
- Stage 1 的原训练/验证 episode 名单如可获得，先排除与质量评估测试集重叠的 episode；若来源无法确认，报告只称“现有 rollout 上的探索性结果”，不声称训练集外泛化。

### 2.2 先定义可观察事件和配对

在仿真真值可用时，结合任务代码定义两类事件的**第一次可核实发生 tick**，阈值引用任务成功判据/碰撞规则并输出实际值，不凭图片主观猜模型内部原因：

| 目标事件 | 当前状态标签（观测时即可计算） | 下一 chunk 风险标签（只作预测目标） |
| --- | --- | --- |
| 中途放置与对齐 | 被持方块相对目标支撑块的 XY 偏移、底面与支撑顶面的 Z/高度差；有必要时角度与支撑重叠 | π0.5 从**本次 policy call**执行的下一段动作内，是否发生放置偏差/不稳/掉落；注明真正事件 tick |
| 放块后回撤 | 夹爪或连杆距应保护方块的最小间隙、当前夹爪开合/方块位置；距离定义与碰撞体说明 | 下一段执行时是否接触、推动或碰倒本应留在桌面/塔上的物体 |

- 核对事件对应的语义阶段、方块编号/堆叠层数、释放与回撤起点；保留 `unknown/other`。相同实例、相同方块与相同阶段附近的成功轨迹作为对照，尽量匹配初始物体位姿及相对进度；**不要求所有成功轨迹和失败轨迹有相同长度**。
- 对每个失败事件，取其发生前最近的实际 policy-call `t`（需要时再取此前 1–2 个调用）；`z_rl(t)` 和 `ref_chunk(t)` 来自此调用的**事前观测**。如果事件在已执行的一个 50-tick chunk 内发生，额外记录距最近 call 的控制 tick；事件之后的 call 只能用作“错误已可见”的正对照，不算提前预警。
- 将两项任务分开：A **当前几何状态的读出**（用 `t` 时刻仿真真值），B **下一段 π0.5 动作的风险预测**（用 `t` 之后发生的事件）。B 是既定 π0.5 轨迹上的预测关联，不是对其他未执行动作的因果评价。
- 一条 episode 可贡献多个调用，但训练/测试必须按 episode 和初始场景分组；相邻帧不能视作独立成功案例。事件没有有效事前 policy call、缺同步图像或几何真值时写清原因，不以未来帧补位。若有逐 tick 仿真状态而没有逐 tick 图像，真值可以标在事件 tick，输入仍必须取事件前实际可用图像。
- 仅人工复核含糊事件与代表样本；无需给每一帧手动贴“错误动作”标签。输出对齐和回撤各自的事件、有效事前样本及可匹配成功样本计数。某类别稀少时不硬算 AUC、不把统计不足说成 token 无信息。

## 3. 实验 A：token 是否真正参与重建

### 3.1 同一 decoder、同一目标、只替换 token

选 170k，必要时同流程做 10k。冻结 VLA、encoder、decoder；优先使用**已核实未用于 Stage 1 训练的同一批观测**及推理时一致的三路图像预处理。若训练数据来源无法核实，继续完成实验并把外推结论标为探索性。保持重建真值、decoder 中来自真实前序 VLA embeddings 的 teacher forcing、损失归一化与 mask 完全不变，仅变动 token：

1. `matched_z`：观测 `i` 的原始 `z_rl(i)`。
2. `within_phase_shuffled_z`：跨**不同 episode**随机换入相同语义阶段、相同方块/堆叠层数的另一观测的 token；有初始布局/相对时间信息时优先进一步匹配。固定多次置换 seed，并保存配对清单。匹配不足的样本不悄悄退化成任意分组。
3. `global_shuffled_z`：跨不同 episode 打乱 token，仅作更容易识别大效应的辅助对照；不同阶段被混合时会产生分布变化，不能作为唯一依据。
4. `constant_z`：零或训练集 token 均值，作为排错辅助；对训练好的 decoder 可能是分布外输入，单独性能差不能证明 token 学到了任务信息。

逐 episode 输出原始重建 MSE/NMSE、cosine 及 `shuffled - matched` 的**配对差**，按独立 episode/instance 做 bootstrap 区间；统计置换重复之间变动。按三路相机、decoder 输出 token 序号/前中后位置及关键阶段细分，查是否只有少量位置依赖 token。若原实现 teacher forcing 不同，记录真实实现并保持每个分组一致。原论文的 autoregressive decoder 会看到真实前序 embedding，所以单看总体重建分数不足以判定 bottleneck 用得有多充分。

可选但更强的 no-token 重建基线：在 Stage 1 **相同的训练集和验证集**上，从头训练不提供 `z_rl`、但保留相同真实前序 embeddings、相近 decoder 容量与预算的模型。单独命名，避免把“训练好的 decoder 接收零 token”称作 no-token 模型。若数据/资源不足，先完成冻结 decoder 的交换实验，把结论限定为“decoder 对 token 是否敏感”。

若有可核实的 Stage 1 **初始化** checkpoint，可用其 encoder 的 token 做后述同一套探针，比较训练前/后；不要把随机初始化 encoder 的 token 直接塞入 170k 的 decoder 来比较重建，那是错配输入。

## 4. 实验 B：错误发生前，token 有没有任务信息

### 4.1 只用轻量 probe，不训练 RL

对 170k 和 10k 使用**完全相同**的样本、时间窗口、分组与 probe 流程；有充分资源时可补一个中间 checkpoint，其他六个先不重跑。评估四类输入：

| 探针输入 | 作用 |
| --- | --- |
| `z_rl` | token 自身可读出多少信息 |
| `robot state` | 本体状态基线 |
| `robot state + ref_chunk` | π0.5 已有输入信息的主基线；`ref_chunk` 必须为当前调用**执行前**的 VLA 提案 |
| `robot state + ref_chunk + z_rl` | 判断 token 在主基线之上是否提供增量信息 |

- `z_rl` 由三路图像的 VLA prefix 提取，当前实现未把机器人状态直接并入该 token；因此 **z-only 弱**不等于 Stage 1 必败。主比较是最后两行的同测试集配对指标差。对各 checkpoint 重用原 rollout 在该调用时记录的同一个 `ref_chunk`；若必须重新采样，固定并记录 VLA 随机条件，避免参考动作变化掺入 token 对比。
- 几何标签优先用仿真世界坐标/碰撞体真值。分别预测 XY 误差、Z/高度差和回撤最小间隙；用 ridge 等低容量回归。报告真实单位 MAE/RMSE、以 episode 等权的结果和按事件/阶段的分层结果。如果环境无这些真值，仅在能定义可信的图像标注时作替代，说明 2D 量与 3D 误差不等价。
- 风险标签限定为**当前 policy call 后下一执行 chunk 内**的真实对齐失败或回撤碰撞；分阶段训练正则化 logistic/线性探针，输出 ROC-AUC、正例稀少时的 PR-AUC、阳性率和校准/阈值曲线。不可把整个失败 episode 的每个早期帧都贴上“当前 chunk 即将失败”。成功前的正常 chunk 也是阴性；错误发生后的调用从该风险测试中移除。
- 另做一个无图像的简单上下文基线：阶段、堆叠层数、起始位置或时间进度（确实在部署时可获得的量分别标记）。用于识别时间/场景捷径；若 `z_rl` 增益只来自与事件同步的显而易见阶段，结论写成阶段关联。
- `ref_chunk` 长度为 50×16，直接展开 800 维可能使小样本对照不稳；统一采用可复现的按关节首/尾/均值/最大变化摘要，或训练折内拟合的正则化/降维。token 与 baseline 各自的 scaler、PCA 和超参数均**只在训练折拟合**，验证折完全隔离。每个输入版本用同类探针和相同有限的正则化搜索，明确特征处理；不要为 full-z 特别调更多参数。

### 4.2 分组验证与定量解释

- 在任何探针训练前冻结 `splits.csv`。优先按初始场景/`instance_id` 分组，并让同一 episode、重复 seed、同一实例的成功/失败配对只出现在同一侧；若可用独立组少，用预设的 GroupKFold/leave-one-instance-out，超参数在训练组内再次分组选择。新随机种子的 rollout 可作为独立最终复核，但不用于调参。
- 报告独立场景组数、成功/失败 episode 数、事件数、每折正负例数。`640 个观测`不是 `640 次独立试验`；按 episode/instance 聚类 bootstrap 给风险 AUC 与主基线之差、几何 MAE 与主基线之差的 95% 区间。不能计算的折标 `NA` 并写原因，不能只挑可计算的好折。
- 对照 `state+ref` 与 `state+ref+z` 在**同样观测**的逐例预测差，画按真实时间距事件的误差/风险曲线及置信区间；给成功对照片段同阶段曲线。风险分数在事件后才变高，只能说明错误可识别，不能说明错误前可预测。
- 预先报告主要假设为“中途放置阶段 XY/Z 精度”和“释放后回撤前的安全间隙/下一 chunk 碰撞风险”；不从十几张图中挑最高的 AUC 宣布成功。区间宽或样本太少时结论为**证据不足**，不能套固定的 AUC>0.8 通过线。

## 5. 可视化与审计样本

不再制作主要用于看整体布局的九 checkpoint PCA/t-SNE 大面板。输出更直观的对照：

1. 重建 `matched_z / within_phase_shuffled_z / global_shuffled_z` 的逐 episode 配对差散点或森林图（独立 episode 为点），以及不同 decoder 位置和任务阶段的误差曲线。
2. 对齐目标：同一 held-out 组上真实 XY/Z 与四种探针预测的图；纵轴标真实物理单位、按 stage/堆叠层分面。
3. 回撤目标：夹爪最小间隙或碰撞风险随实际 policy-call 时间变化的图；画出“最后一个事件前 call”“实际事件 tick”“事件后首个 call”的位置。若无同期观测，图中保留时间空白，不连成虚假的 30 Hz token 曲线。
4. 对齐与回撤各至少选 3 条成功和 3 条失败（不足就全取）的多相机关键帧拼图；标 episode、初始 seed、阶段、chunk 起止、真实几何值、探针得分和失败时刻。选择规则预先固定，不能只挑图最好看的例子。

所有图都链接或列出底层样本表与生成命令；图上醒目标注“误差发生前/后”“几何真值/预测”和样本单位。

## 6. 判定口径、执行终点与产物

结果按下表给出，而不是强行选“最佳 ckpt”：

| 观察 | 结论及下一步 |
| --- | --- |
| 真实 token 重建明显优于同阶段交换 token；pre-action full-z 也稳定优于 `state+ref` | 有证据支持 token 被重建使用，且保留当前任务关键状态/风险信息；以后可以**另行**进入 Actor/Critic 方案，但本轮不执行 |
| 重建依赖 token，但对齐/回撤关键 probe 无增量 | token 可能保留了一般视觉信息；不能认定已有证据足够支撑三积木关键行为。查相机可见性、token 训练数据覆盖、输入时序及对齐/回撤事件条件 |
| 交换 token 对重建影响极小 | 先核查置换样本、teacher forcing、缓存与 decoder 接线；no-token 对照可判断前序 embedding 是否足以独立重建。不要仅继续堆 Stage 1 step 或直接启动 RL |
| 事件发生在 chunk 内、事前图像看不到风险，或有效事件/独立场景太少 | 当前数据/50 步观测节奏下**无法判断**是否能提前预测；需要更密的观测日志或定向收集失败场景，不把事件后的 AUC 当预警性能 |

不设通用 NMSE、cosine 或 AUC 阈值。重要结果是**同数据、同时间语义、独立场景下的配对增量**与不确定性；10k vs 170k 的差别也要用同样口径。若只是可见失败之后有信号，报告必须写“事后识别”。

在当前工程建立独立输出目录，例如 `runs/rlt_token_quality_validation/<run_id>/`，不得覆盖旧 replay、旧图表或旧 Actor 权重：

```text
run_manifest.yaml                 # 代码/数据/两 ckpt 身份、相机预处理与真值规则
episodes.csv
calls.csv
events.csv                       # 对齐、释放、回撤事件 tick 和失败类型
splits.csv                       # 分组、场景/seed、训练验证划分及样本数
reconstruction_controls.csv      # 每观测/episode matched、置换、常量对照
reconstruction_summary.csv       # episode 等权差异、区间、阶段/位置统计
probe_predictions.csv            # 各输入版本在各 held-out fold 上的逐例预测
probe_metrics.csv                # 当前几何与下一 chunk 风险，含差值和区间
figures/                         # 配对差、几何预测、事件时间线、关键帧
commands.txt                     # 已实际执行的命令和失败/恢复记录
REPORT.md                        # 中文结果：承载信息、关键事前信息、增量价值
```

Codex 应完成能运行的诊断代码与实验；若缺真实几何/事件标签，先按可用的仿真状态和任务规则实现、核验部分示例，再报告无法计算的项目。不要造标签、拿整条失败 episode 代替下一 chunk 风险，或在结尾自动启动 Actor/Critic。最终报告明确列出哪些检验完成、哪些由于数据不足尚不可判，并给出最省成本的下一批观测需求。

## 7. 方法依据

- [RL Token 原论文，第 IV-A 节](https://arxiv.org/html/2604.23073)：Stage 1 的 token 与 decoder 重建目标；decoder 输入包含真实前序 embedding，本轮据此设计 token 交换对照。
- [openpi-RLT 公开仓库](https://github.com/Yyshadow/openpi-RLT)：本地模型/服务路径的定位线索；Stage 1 对照以用户本地代码为准。
- [之前的复现笔记](https://villekuosmanen.medium.com/research-notes-from-reproducing-rl-token-f375ecfd3c28)：成功/失败 token 图是探索性例子；本轮检验针对用户当前任务的事前信息和增量价值。

置换次数、分组探针、bootstrap 和三积木的几何目标是本轮实验设计，不是论文指定的官方 Stage 1 验收标准。