# RLT 复现指南

停止目前本机正在进行的训练，然后从头重新进行 RL Token \+ Actor/Critic 的完整离线复现实验。

这次实验需要能够无人值守连续运行。

总体流程：

```Plain Text
停止并保存当前实验
        ↓
Stage 1：RL Token 训练到 20k
        ↓
自动评估所有 Stage-1 checkpoints
        ↓
自动选择最佳/充分收敛的 RL Token checkpoint
        ↓
Stage 2：加载现有 50 success + 100 failure rollout
        ↓
构建/检查 replay
        ↓
Critic + Actor 离线训练
        ↓
Critic 10k / 20k / 30k 多 checkpoint 验证
Actor约 5k / 10k / 15k
        ↓
自动生成诊断结果、曲线和 summary
        ↓
停止
```

---

## 停止当前训练

---

# Stage 1：重新训练 RL Token

## Base VLA

必须使用我已经 SFT\-trained 的 π0\.5。

检查并打印最终实际使用的：

```Plain Text
VLA config
SFT checkpoint
dataset
DataConfig
action normalization
state normalization
image transforms
action horizon
action dimension
batch size
learning rate
RL token num
RL token layers
RL token dimension
rlt_alpha
```

不要擅自改变已经跑通的数据预处理。

---

## Stage\-1 数据

Stage 1 继续使用原来 SFT π0\.5 的 demonstration dataset。

目前已有：

```Plain Text
50 success rollout
100 failure rollout
```

这 150 条数据留给 Stage 2。

Stage 1 数据流保持：

```Plain Text
demonstration observation
        ↓
Frozen SFT π0.5
        ↓
VLA internal representation h
        ↓
RL Token Encoder
        ↓
z_rl
        ↓
RL Token Decoder
        ↓
reconstructed h_hat
```

---

## 冻结 VLA

Stage 1 使用：

```Plain Text
rlt_alpha = 0
```

原则：

```Plain Text
SFT π0.5：frozen

RL Token encoder：trainable
RL Token decoder：trainable
```

不要只根据 config 名字假设冻结成功。

实际检查 optimizer / parameter tree。

训练启动时打印：

```Plain Text
Trainable parameters
Frozen parameters
Total parameter count
Trainable parameter count
```

确认只有 RL\-token 模块被更新。

如果仓库源码中 `rlt_alpha=0` 的真实语义与这里不同，以源码为准，同时在报告中说明。

---

# Stage 1 一次连续训练到 20k

不要分别独立训练 5k / 10k / 20k。

使用一次连续训练：

```Plain Text
0 → 20,000 steps
```

保持：

```Plain Text
seed
dataset
batch size
learning rate
optimizer
augmentation
model architecture
```

完全一致。

checkpoint：

```Plain Text
save_interval = 2500
```

至少永久保留：

```Plain Text
5k
10k
15k
20k
```

注意检查 checkpoint cleanup / keep\_period。

不要让这些关键 checkpoint 被自动删除。

---

# 固定 Validation Set

如果当前 Stage\-1 没有真正独立的 validation set，则新增。

必须按：

```Plain Text
episode
```

划分。

禁止同一个 episode 中的不同 frames 同时进入 train 和 validation。

优先使用已有官方 split。

否则：

```Plain Text
90% episodes → train
10% episodes → validation
```

固定 random seed。

validation dataset 从开始到结束保持完全不变。

---

# Stage\-1 Metrics

不能只记录：

```Plain Text
rlt_loss
```

至少记录：

### Reconstruction MSE

```Plain Text
train_reconstruction_mse
val_reconstruction_mse
```

### Embedding variance

目标 embedding：

```Plain Text
h
```

计算：

```Plain Text
embedding_variance
```

必须和 reconstruction 使用相同 valid mask。

### Normalized MSE

```Plain Text
val_nmse =
val_reconstruction_mse / embedding_variance
```

### R²

```Plain Text
R2 =
1 - SSE / SST
```

输出：

```Plain Text
val_r2
```

### Cosine Similarity

输出：

```Plain Text
val_cosine_mean
val_cosine_std
val_cosine_p10
val_cosine_p50
val_cosine_p90
```

### Representation stability

记录：

```Plain Text
mean ||z_rl||
std ||z_rl||

mean per-dimension std
```

检查：

```Plain Text
NaN
Inf
representation collapse
abnormal norm explosion
```

---

# 自动评估 Stage\-1 checkpoints

训练到 20k 后，不要等待人工确认。

自动评估：

```Plain Text
5k
10k
15k
20k
```

如果成本很低，也可以评估所有 interval checkpoint。

保存：

```Plain Text
analysis/stage1_checkpoint_metrics.csv
```

至少：

```Plain Text
step
train_mse
val_mse
val_nmse
val_r2
val_cosine
z_norm_mean
z_norm_std
```

---

# Stage\-1 自动 checkpoint 选择规则

这是无人值守实验最重要的一部分。

不能简单：

```Plain Text
选择 val MSE 最低的 checkpoint
```

也不能默认：

```Plain Text
选择 20k
```

采用下面的保守规则。

## 第一步：淘汰异常 checkpoint

出现任何以下情况直接排除：

```Plain Text
NaN
Inf
representation collapse
明显 z_rl norm explosion
validation metric 无法计算
checkpoint 损坏
```

---

## 第二步：确定最优 validation NMSE

在有效 checkpoints 中找到：

```Plain Text
best_nmse
```

然后建立候选集合：

```Plain Text
val_nmse <= best_nmse * 1.03
```

即 NMSE 距离全局最好值不超过约 3%。

---

## 第三步：检查 R² 和 cosine

候选 checkpoint 同时要求：

```Plain Text
R² 没有明显差于最佳 checkpoint

cosine similarity 没有明显差于最佳 checkpoint
```

可使用：

```Plain Text
R² 与 best 相差 <= 0.01

cosine 与 best 相差 <= 0.005
```

如果这些阈值因实际数值尺度明显不合理，可以只用于 tie\-break，而不要强行导致无 checkpoint 可选。

---

## 第四步：优先选择更早充分收敛的 checkpoint

如果：

```Plain Text
10k
15k
20k
```

性能已经非常接近，

例如：

```Plain Text
10k NMSE = 0.101
15k NMSE = 0.099
20k NMSE = 0.098
```

不要因为 20k 数值略低就一定选择 20k。

选择：

```Plain Text
最早进入“接近最佳性能区域”的 checkpoint
```

这样减少过拟合风险，也保持模型选择原则简单。

换句话说：

> 在性能基本等价的 checkpoints 中选择更早的那个。
> 
> 

---

## 第五步：fallback

如果上述条件无法得到候选：

选择：

```Plain Text
validation NMSE 最低
```

的有效 checkpoint。

tie\-break：

```Plain Text
higher R²
→ higher cosine
→ earlier checkpoint
```

---

# Stage\-1 自动安全门

如果出现：

```Plain Text
所有 checkpoint 无效
```

或者：

```Plain Text
NaN / Inf
严重 representation collapse
validation loss 完全发散
```

则：

```Plain Text
DO NOT START STAGE 2
```

写：

```Plain Text
FAILED_STAGE1.md
```

保存错误日志后停止。

不要为了让流程继续而自动修改 learning rate、architecture 或其他超参数。

---

# Stage\-1 成功后自动进入 Stage 2

如果 Stage\-1 validation 正常：

把自动选择出的：

```Plain Text
best_stage1_checkpoint
```

写入：

```Plain Text
analysis/selected_stage1_checkpoint.txt
```

同时记录：

```Plain Text
step
val MSE
NMSE
R²
cosine similarity
选择原因
```

然后自动启动 Stage 2。

不需要等待人工确认。

---

# Stage 2：Actor \+ Critic Offline Training

## 使用现有 rollout

我目前已有：

```Plain Text
50 successful episodes
100 failed episodes
```

Stage 2 使用这些已有 rollout。

先检查每条 episode 是否能够恢复：

```Plain Text
observation_t

camera observations

proprio / robot state

executed action_t

next observation

episode boundary

success / failure

done
```

如果已经有标准 replay journal，则直接检查并复用。

如果没有，则按照当前 openpi\-RLT 仓库的数据结构转换。

不要自己创造不存在的数据字段。

---

# 使用 selected Stage\-1 model 生成 RLT features

对 replay 中 observation 使用选中的：

```Plain Text
best_stage1_checkpoint
```

生成：

```Plain Text
z_rl
proprio
ref_chunk
```

最终 transition 至少应包含：

```Plain Text
curr:
    z_rl
    proprio
    ref_chunk

action:
    actually executed action

reward

done

next:
    next_z_rl
    next_proprio
    next_ref_chunk

episode_id
step_id
success
```

确保：

```Plain Text
ref_chunk
```

来自冻结的 SFT π0\.5。

---

# Reward 第一版保持简单

如果目前仓库已经使用 sparse terminal reward，则不要重新设计 reward。

保持：

```Plain Text
success episode:
0
0
...
1

failure episode:
0
0
...
0
```

---

# Stage\-2 Train / Validation split

必须按 episode split。

不能把一个 episode 的 transition 分散到 train 和 validation。

因为共有：

```Plain Text
50 success
100 failure
```

请做 stratified episode split。

例如：

```Plain Text
80% train
20% validation
```

大约：

```Plain Text
train:
40 success
80 failure

validation:
10 success
20 failure
```

如果当前项目已有固定 split，优先复用。

固定随机 seed。

记录具体 validation episode IDs。

---

# Stage\-2 参数原则

第一轮不要乱调算法参数。

沿用当前 openpi\-RLT 已经跑通的默认配置，包括类似：

```Plain Text
actor_lr
critic_lr
gamma
target_tau
BC weight
Q weight
reference dropout
batch size
actor update period
```

如果当前实际 config 与仓库 README 不同，以当前仓库源码/config 为准，并在报告中打印真实值。

只扩大训练预算。

---

# Stage\-2 训练预算

本轮训练到：

```Plain Text
critic updates = 30,000
```

如果当前实现：

```Plain Text
actor_update_period = 2
```

则 actor 大约：

```Plain Text
15,000 updates
```

不要为了恰好得到 15k actor update 而改变原更新比例。

重点保存：

```Plain Text
Critic 10k
Actor ~5k

Critic 20k
Actor ~10k

Critic 30k
Actor ~15k
```

最好每：

```Plain Text
1000 critic updates
```

保存一次 metrics。

主要 checkpoint 至少永久保留：

```Plain Text
C10k/A5k
C20k/A10k
C30k/A15k
```

---

# Critic 必须增加 validation diagnostics

不要只记录：

```Plain Text
critic train loss
```

计算 validation replay 的真实 Monte\-Carlo discounted return：

```Plain Text
G_t =
r_t +
gamma r_(t+1) +
...
```

然后比较：

```Plain Text
Q(s_t, a_t)
```

和：

```Plain Text
G_t
```

至少记录：

```Plain Text
critic_val_td_loss

Q_vs_MC_MSE

Q_vs_MC_Pearson

Q_vs_MC_Spearman
```

---

# Success / Failure Q 分离

分别计算：

```Plain Text
Q_success_mean
Q_success_median

Q_failure_mean
Q_failure_median
```

记录：

```Plain Text
Q_success_minus_failure
```

这一项非常重要。

目标不是硬编码要求某个绝对数值，而是观察 critic 是否逐渐形成有意义的 return ordering。

---

# Twin Critic 诊断

如果当前实现是 twin\-Q：

记录：

```Plain Text
Q1
Q2
```

以及：

```Plain Text
mean |Q1 - Q2|
median |Q1 - Q2|
p90 |Q1 - Q2|
```

用于发现 critic instability / disagreement。

---

# Actor diagnostics

Actor 不允许只看 action MSE。

同时输出：

```Plain Text
actor_val_action_mse
```

以及 actor 相对于 VLA reference action 的修改：

```Plain Text
delta_action =
actor_action - ref_action
```

统计：

```Plain Text
mean |delta|
median |delta|
p90 |delta|
max |delta|
```

并分别统计每个 action dimension。

如果动作是 normalized space：

同时尽量输出反归一化后的：

```Plain Text
per-joint MAE
per-joint RMSE
```

使用实际动作单位。

---

# Actor 是否真的利用 Critic

在 validation samples 上比较：

```Plain Text
Q(s, actor_action)
```

和：

```Plain Text
Q(s, ref_action)
```

记录：

```Plain Text
Q_actor_mean
Q_ref_mean
Q_actor_minus_ref
```

这里主要用作诊断。

不要为了最大化：

```Plain Text
Q_actor_minus_ref
```

自动疯狂训练 actor。

如果 actor 对 reference 做极端偏移，而 critic Q 却异常升高，需要标记为可能：

```Plain Text
critic exploitation
OOD action
```

而不是判断为性能很好。

---

# Stage\-2 checkpoints 自动比较

训练结束后统一比较：

```Plain Text
C10k/A5k
C20k/A10k
C30k/A15k
```

表格至少包含：

```Plain Text
checkpoint

critic_val_td_loss

Q_MC_Pearson
Q_MC_Spearman

Q_success
Q_failure
Q_success_minus_failure

Q1_Q2_gap

actor_action_MSE

actor_delta_mean
actor_delta_p90

Q_actor
Q_ref
Q_actor_minus_ref
```

---

# Stage\-2 checkpoint 推荐规则

不要使用：

```Plain Text
最低 actor MSE
```

直接作为最佳 Actor/Critic checkpoint。

首先淘汰：

```Plain Text
NaN / Inf

critic divergence

Q 值明显爆炸

Twin-Q disagreement 持续异常增长

actor delta 明显异常爆炸
```

然后主要参考：

```Plain Text
Q vs Monte-Carlo Spearman correlation
Q vs Monte-Carlo Pearson correlation
critic validation TD loss
success/failure Q separation
Twin-Q consistency
actor delta stability
```

优先选择 critic 对真实 return 排序最好、且 actor 没有发生异常偏移的 checkpoint。

如果：

```Plain Text
20k
30k
```

表现基本相同，

优先推荐：

```Plain Text
20k
```

即在性能近似时选择较早 checkpoint。

不要仅因为训练到了 30k 就自动认为 30k 最好。

最终将推荐 checkpoint 写入：

```Plain Text
analysis/selected_stage2_checkpoint.txt
```

---

# 自动生成曲线

Stage 1：

```Plain Text
stage1_train_val_mse.png
stage1_nmse.png
stage1_r2.png
stage1_cosine.png
stage1_z_norm.png
```

Stage 2：

```Plain Text
critic_train_val_loss.png

q_mc_correlation.png

q_success_failure.png

twin_q_gap.png

actor_action_mse.png

actor_reference_delta.png

q_actor_vs_reference.png
```

---

# 保存 CSV

至少生成：

```Plain Text
analysis/stage1_checkpoint_metrics.csv

analysis/stage2_training_metrics.csv

analysis/stage2_checkpoint_comparison.csv
```

这样后续我可以直接分析。

---

# 最终 summary

所有任务完成后自动生成：

```Plain Text
analysis/final_summary.md
```

必须包含：

```Plain Text
=============================
STAGE 1
=============================

SFT checkpoint:
dataset:
train episodes:
validation episodes:

RL Token architecture:
training steps:

5k:
val MSE =
NMSE =
R2 =
cosine =

10k:
...

15k:
...

20k:
...

Selected Stage-1 checkpoint:
Selection reason:

Representation collapse:
yes/no

Overfitting:
yes/no

Plateau:
yes/no


=============================
STAGE 2
=============================

Replay episodes:
success =
failure =

Replay transitions:

Train episodes:
Validation episodes:

Actor config:
Critic config:

C10k/A~5k:
TD loss =
Q-MC Pearson =
Q-MC Spearman =
Q success =
Q failure =
Twin-Q gap =
Actor MSE =
Actor delta =

C20k/A~10k:
...

C30k/A~15k:
...

Recommended Stage-2 checkpoint:
Reason:


=============================
DIAGNOSIS
=============================

Stage 1:
healthy / suspicious / failed

Critic:
healthy / undertrained / overfit / unstable

Actor:
healthy / copying reference / excessive deviation / unstable

Recommended next action:
```

最后的：

```Plain Text
Recommended next action
```

只能根据已经观察到的数据给出，例如：

```Plain Text
Stage-1 10k 后已经平台，继续训练意义不大

Critic 10k→20k 提升明显，20k→30k基本不变

Actor delta 稳定

建议下一步使用 C20k/A10k 做在线评估
```

不要自动修改参数然后继续第三轮训练。

---

# 故障处理

整个任务是无人值守执行，因此遇到可恢复问题：

```Plain Text
临时 evaluation script bug
路径错误
checkpoint naming问题
CSV绘图问题
```

可以检查原因并修复后继续。

但是不要在以下情况自动改变实验设计：

```Plain Text
loss不好
validation指标不好
critic没收敛
actor MSE偏高
```

也就是说：

**代码错误可以修。**

**实验结果不好不能偷偷调参。**

如果 Stage 1 本身训练失败：

不要进入 Stage 2。

如果 Stage 2 某个 checkpoint evaluation script 出错：

保留已经训练好的 checkpoint，修复 evaluation 后继续评估。

---

# 所有源码修改必须记录

如果必须修改仓库源码：

保持修改最小化。

保存：

```Plain Text
git diff
```

最终输出：

```Plain Text
analysis/code_changes.diff
```

并在 final\_summary 中列出：

```Plain Text
Modified files:
Reason:
```

不要删除原实现。

诊断脚本尽量新增独立文件。

---

# 最终执行边界

今晚自动执行允许：

```Plain Text
Stage-1 RL Token training
Stage-1 validation
Stage-1 automatic checkpoint selection

replay preprocessing

Stage-2 offline Actor/Critic training
Stage-2 offline validation
checkpoint comparison
plotting
summary generation
```

今晚禁止自动执行：

```Plain Text
真实机器人动作
无人看守机器人 rollout
新的 physical data collection
自动 online robot RL
```

Stage\-2 offline 完成、保存结果和 summary 后：

```Plain Text
STOP
```

不要等待人工输入，也不要因为实验结果不好自动开始新的 hyperparameter sweep。

本轮核心目标是得到一条完整、可解释的：

```Plain Text
SFT π0.5
   ↓
RL Token
   ↓
自动选 Stage-1 checkpoint
   ↓
已有150 episodes
   ↓
Actor/Critic offline training
   ↓
10k / 20k / 30k comparison
   ↓
recommended Stage-2 checkpoint
```

并让我之后能够明确判断：

```Plain Text
问题出在 RL-token representation

还是 Critic

还是 Actor

还是应该进入真正的 online rollout 验证。
```

