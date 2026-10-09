# RL Token 多 checkpoint 特征可视化与对比：Codex 执行文档

## 目标

基于已经训练好的 openpi-RLT RL token checkpoint，对多个训练 step 的
checkpoint 进行离线特征分析，回答：

1.  不同训练阶段的 RL token (`z_rl`) 是否发生语义变化；
2.  RL token 是否能够编码成功/失败、关键动作阶段的信息；
3.  哪一个 checkpoint 更适合作为后续 Actor-Critic 或进一步实验的输入。

本任务只做冻结模型分析，不重新训练 RL token、Actor、Critic，不改变
rollout 配置。

RL token 在 RLT 中作为 VLA 内部表示到轻量 RL
模型之间的接口，因此分析重点是其表示质量，而不是二维图本身是否"分开"。
citeturn0search2turn0search3

------------------------------------------------------------------------

# 1. 已知环境

## 模型

-   Base VLA: `/mnt/pfs/kk/kk/ckpt/geniesim3/spatialpi05`

-   RL token checkpoints:

```
    /mnt/pfs/kk/kk/ckpt/RLT/geniesim_stack_three_blocks_plan/checkpoints/rlt_pi05_geniesim_stack_three_blocks_plan/stage1_rl_token_20k/
```

候选 checkpoint:

    170k
    150k
    130k
    110k
    90k
    70k
    50k
    30k
    10k

## 数据

Rollout:

    /mnt/pfs/kk/kk/data/data/geniesim/rollout

已有：

-   成功 rollout：约 50
-   失败 rollout：约 50

保存：

-   RGB 图像
-   robot state
-   episode 信息

目前没有保存 `z_rl`，需要离线重新提取。

------------------------------------------------------------------------

# 2. 核心修改：checkpoint 对比方式

## 原则

不要把不同 checkpoint 的 token 混合训练 PCA。

原因：

不同 checkpoint 的 latent space 可能发生旋转、尺度变化。

因此：

每一个 checkpoint 单独：

    checkpoint
        ↓
    固定 rollout 子集
        ↓
    提取 z_rl
        ↓
    PCA/t-SNE
        ↓
    保存结果

同时增加一个：

## 跨 checkpoint 数值比较

对于每个 checkpoint 输出：

-   PCA explained variance
-   temporal smoothness
-   success/failure separability
-   reconstruction metric
-   cosine similarity
-   token norm statistics

------------------------------------------------------------------------

# 3. 数据子集固定（重要）

所有 checkpoint 必须使用完全相同的数据。

保存：

    episodes_manifest.csv

包含：

-   episode_id
-   success
-   failure
-   seed
-   frame indices

禁止：

-   不同 checkpoint 使用不同 episode
-   根据图效果挑样本

------------------------------------------------------------------------

# 4. z_rl 提取

必须确认：

提取的是：

    RL token encoder output
            |
            ↓
          z_rl

不是：

-   learned token embedding
-   VLM hidden state
-   decoder feature
-   action embedding

保存：

    features/
        ckpt_10k/
            features.npz

        ckpt_30k/
            features.npz
    ...

每个：

    z:
    [N,D]

    metadata:
    sample_id
    episode_id
    frame_id
    timestamp
    success

------------------------------------------------------------------------

# 5. 可视化设计

## 5.1 单 checkpoint 内部分析

每个 checkpoint 输出：

### PCA

固定：

    PCA fit(all samples)

输出：

1.  success/failure

观察：

-   是否失败轨迹偏离成功轨迹
-   差异出现在哪个阶段

2.  时间颜色

颜色表示：

    episode normalized time

3.  trajectory

连接：

同一个 episode 的 token：

    z1 → z2 → z3 → ...

不要跨 episode 连线。

------------------------------------------------------------------------

## 5.2 t-SNE

用途：

辅助观察局部结构。

参数固定：

    random_state=42
    init=pca
    perplexity=30

不同 checkpoint 使用同样参数。

注意：

不能因为某个 checkpoint t-SNE 分得更开，就认为一定更好。

------------------------------------------------------------------------

# 6. 新增 checkpoint 横向比较

新增：

    checkpoint_summary.csv

格式：

  -------------------------------------------------------------------------------------------------------
  checkpoint   D       N       PCA_var1   PCA_var2   success_gap   smoothness   reconstruction   cosine
  ------------ ------- ------- ---------- ---------- ------------- ------------ ---------------- --------
  10k                                                                                            

  30k                                                                                            

  50k                                                                                            

  ...                                                                                            
  -------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 7. 新增核心指标

## 7.1 Success / Failure 分离

不要只看图片。

计算：

### centroid distance

成功 token:

    μ_success

失败 token:

    μ_failure

计算：

    ||μ_success-μ_failure||

------------------------------------------------------------------------

## 7.2 temporal smoothness

同一 episode：

    distance(z_t,z_t+1)

统计：

mean/std

用于判断 token 是否随着任务状态连续变化。

------------------------------------------------------------------------

## 7.3 phase sensitivity

如果已有 stage:

计算：

不同阶段：

    grasp
    transport
    align
    release
    retract

token distance。

如果没有：

只使用时间。

------------------------------------------------------------------------

# 8. checkpoint 选择逻辑

自动生成：

    checkpoint_ranking.md

但是：

不要输出"最佳 checkpoint"。

改成：

例如：

-   checkpoint A reconstruction 最好
-   checkpoint B success/failure 分离更明显
-   checkpoint C temporal smoothness 更稳定

最终由研究者决定后续使用。

------------------------------------------------------------------------

# 9. 运行流程

## Stage 0

检查：

-   backbone 是否一致
-   RL token 是否加载成功
-   missing/unexpected keys

保存：

    resolved_config.yaml

------------------------------------------------------------------------

## Stage 1

4 episode 测试：

    2 success
    2 failure

确认：

-   z_rl 正常
-   图像对应
-   无 NaN

------------------------------------------------------------------------

## Stage 2

正式：

    20 success
    20 failure

每个 checkpoint 重复。

------------------------------------------------------------------------

## Stage 3

生成：

    rlt_token_analysis/

    ├── episodes.csv
    ├── frames.csv

    ├── ckpt_10k/
    │   ├── features.npz
    │   ├── pca.png
    │   ├── tsne.png

    ├── ckpt_30k/
    ...

    ├── checkpoint_summary.csv

    ├── checkpoint_ranking.md

    └── report.md

------------------------------------------------------------------------

# 10. 最终报告要求

report.md 必须回答：

1.  哪些 checkpoint 的 token 表示发生明显变化？

2.  reconstruction 提升是否对应 representation 更好？

3.  成功/失败是否在 token 空间出现差异？

4.  差异发生在哪个任务阶段？

5.  是否存在某些 checkpoint：

    -   reconstruction 好
    -   但是 token 对任务状态不敏感？

------------------------------------------------------------------------

# 11. 重要限制

不要下结论：

"t-SNE 分开，所以 RL 一定有效"。

正确表述：

"该 checkpoint 的 RL token 在当前 rollout
分布下包含与成功/失败相关的信息"。

最终需要结合：

-   Actor-Critic 性能
-   rollout success rate
-   downstream RL

进行验证。

------------------------------------------------------------------------

# Codex执行要求
停止目前本机多卡的训练，后续的执行你可以自定义调用本机8张gpu，自定义显存使用，力求高效率得到实验结果

具体执行放到tmux窗口中，防止ssh突然断开连接导致程序停止

然后请直接执行：

1.  检查工程和接口；
2.  完成 z_rl offline extraction；
3.  固定数据集；
4.  对 9 个 checkpoint 批量分析；
5.  保存所有中间结果；
6.  输出中文 report.md。


不要只输出代码。