我需要修正前面的两个判断：

  1. 现在不能简单归因于“Critic 训练步数不够”。更准确地说，是 Critic 获得的*数据分布*、*时序定义*和*动作维度*可能不适合学习可靠的动作排序。
  2. “Replay 中需要同一状态下多个动作”表述不准确。RLT 不要求同一个状态输入多个 ref_action。每次只需要 VLA 给出的一个 reference action，关键是后续在线阶段要积累 Actor 实际执行的
     不同动作及结果。

  你现在的 150 条轨迹已经足够作为 warmup 数据，暂时没必要继续采集同类型的 π0.5 数据。

  关于 20k 是否过拟合

  20k 不能算已经明确过拟合：

  - 20k：validation Pearson 0.5822，Spearman 0.1413
  - 30k：validation Pearson 0.6015，Spearman 0.1421
  - 最好 validation Spearman 大约在 27k：0.1658

  因此现象更接近：

  - Critic 很快拟合训练集；
  - validation Pearson 还在缓慢增加；
  - 但真正重要的动作价值排序 Spearman 很低，而且很早饱和。

  你的 replay 有 4122 个 transition，20k 次更新对应约 4.85 updates/transition，恰好接近论文的 UTD=5。所以问题不是“20k 步太多”，而是 固定数据集持续更新无法提供新的动作结果信息。

  50 条数据为什么可能够

  论文中提到的 50 条 base-policy rollout，主要出现在基线或 warmup 设置中，并不表示“50 条离线轨迹就能完成整个 RLT”。正式 RLT 流程是：

  1. VLA warmup 收集数据；
  2. 训练初始 Critic 和 Actor；
  3. 用 Actor 继续在线执行；
  4. 将 Actor action、reference action、reward 和人工干预写回 replay；
  5. 继续训练并循环。

  这是论文明确描述的闭环流程。RLT paper，official project page

  你的 150 条数据包含 50 成功、100 失败，共 4122 个 transition，但导出时：

  - ref_action 来自记录的 π0.5 action；
  - executed action 被直接设置为 ref_action；
  - 所有数据都是 BASE/warmup。

  相关逻辑在 RLT/scripts/geniesim/export_rollout_replay.py:121。因此 150 条足够初始化 Critic，却不能替代 Actor 在线探索数据。继续增加相似的 π0.5 rollout，边际收益可能很小。

  Reference action 只有一个是否有问题

  没有问题。RLT 本身就是每个 observation 由 VLA 给一个 reference action chunk。所需的动作差异来自不同时间的在线交互：

  状态 s
    -> π0.5 给 reference action
    -> Actor 根据 s、RL token、reference 生成修正动作
    -> 环境返回结果
    -> 新 transition 写入 replay

  不需要机器人精确回到同一个状态尝试多个动作。Critic 可以从相近状态、不同 Actor 版本、不同失败扰动中学习动作和结果的关系。

  但你当前 replay 中没有 Actor 执行动作，Critic 很容易只根据视觉状态判断“这一局像成功还是失败”，而不真正使用动作。这会产生较高 Pearson，却不能可靠指导 Actor。

  Actor 为什么几乎复制 π0.5

  当前 20k 配置中，Actor 各项损失大致为：

  - 加权 BC：0.0462
  - 加权 residual/delta：0.0212
  - 加权 Q：0.00174

  从数值看，模仿与残差约束合计约为 Q 项的 39 倍。损失值不等于梯度大小，但已经说明 Q 信号很难推动 Actor 明显改变动作。

  这不是 RLT 设计错误。论文有意让 Actor 靠近 VLA，避免弱 Critic 被利用。问题在于你的训练一直使用 warmup 权重 BC=10，而没有进入在线阶段的 BC=5，同时 Critic 对动作的排序能力还弱。
  此时直接减小 BC 或增大 Q 权重，最可能得到的就是动作抖动，而不是性能提升。

  另外，当前训练使用：

  - action chunk C=50
  - action dimension 16
  - Critic 一次接收 800 维动作
  - exploration std 0.05

  而论文和仓库当前 Ethernet 配置使用 C=10；仓库配置的 std 是 0.002，见 RLT/rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml:4。不同任务不能直接照搬参数，但 0.05 对归一
  化机器人动作可能明显过大，是抖动的重要嫌疑。

  建议的改进顺序

  1. 先确认真实执行长度

     查清每次 π0.5 预测 50 步后，控制器实际执行的是 10、20 还是 50 步。TD target 当前使用 gamma^50，见 RLT/scripts/geniesim/train_stage2_plan.py:231。如果实际只执行前 10 步，却
     按 50 步折扣，这是确定的时序错误。

  2. 把 Actor/Critic chunk 改为实际执行前缀

     建议首先实验 C=10。这会把动作输入从 800 维降到 160 维，更容易让 Critic 学到动作差异，也符合论文采用短 Actor chunk 提高反应性的设计。

  3. 降低探索方差

     对 fixed_std 做 0.002 / 0.005 / 0.01 三档实验，暂时不要继续用 0.05。部署时先使用 deterministic mean action，并限制每个关节的最大 residual。

  4. 验证 Critic 是否真的使用动作

     在 validation 上增加：
      - Q(s, a) 对比 Q(s, shuffled_a)
      - 对动作逐渐加噪后的 Q 曲线
      - ||∂Q/∂a||
      - 仅输入 state 的 baseline
      - 同一状态下 reference 与 Actor action 的 Q margin

     如果打乱动作后 Q 几乎不变，说明 Critic 主要在识别状态或 episode 结果，不能用于优化 Actor。

  5. 不要再无限延长固定数据训练

     30k 后 Spearman 没有稳定提高，继续到 100k 大概率只会让训练集拟合得更好。用 validation Spearman、action sensitivity 和真实成功率选择 checkpoint，而不是训练 loss。

  6. 启动小批量在线闭环

     用现有 150 条作为 warmup。先训练较保守的 Actor，然后每轮收集 5–10 条 Actor rollout，记录：
      - π0.5 reference action
      - Actor 实际执行 action
      - reward
      - 是否干预
      - Actor 与 reference 的差值

     每轮追加 replay 后再更新网络。最初设置 residual 上限和 π0.5 fallback，避免异常动作直接扩大。

  7. 进入在线数据后再减弱 BC

     warmup 可维持 BC=10, Q=0.1；有稳定 Actor 数据后切换到仓库已有的 BC=5, Q=0.1。同时记录每个 loss 项对应的梯度范数，再决定是否增加 Q 权重。不要仅根据 loss 数值调比例。

  8. 把 RLT 聚焦到失败关键阶段

     你现在训练的是完整“三块堆叠”，平均约 27.5 次 policy call 才得到终局结果，信用分配很困难。更合适的是先针对抓取对齐、接触和放置等高失败阶段训练 RLT，或者为完整任务增加可靠的
     子目标 reward。官方实验同样强调对任务关键阶段进行在线改进。

  所以，我对你当前情况的判断是：不是数据总条数不足，也不是 Critic 单纯少训练了几万步；主要问题是目前只完成了离线 warmup，Critic 缺少 Actor 分布的数据，同时 C=50、折扣时序和
  std=0.05 都需要优先核查。 在这些问题解决前，通过降低 BC 强迫 Actor 偏离 π0.5，反而会放大抖动。