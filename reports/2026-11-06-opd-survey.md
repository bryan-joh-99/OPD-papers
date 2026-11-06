# OPD 调研报告

> On-Policy Distillation（OPD）最新论文与开源实现跟踪

**报告版本：2026-11-06**
**检索范围：** arXiv、论文附带的项目页和 GitHub 代码仓库
**说明：** 下文的实验数字均为论文摘要或论文报告的结果，不代表本仓库已经复现。除特别注明外，论文均为 arXiv 预印本。

## 一、结论摘要

OPD 的基本做法是让 student 在自己的 rollout 上生成训练状态，再由更强的 teacher 在这些状态上提供 token-level 分布或分数。它比固定 teacher 轨迹上的 KD 更贴近部署时分布，尤其适合 reasoning、agent 和 speculative decoding 场景。

截至本报告日期，研究重点已经从“是否使用 on-policy 数据”转向四个更具体的问题：

1. **训练哪些状态：** student-only rollout 会遗漏 teacher 能到达、但 student 尚未到达的状态。
2. **使用什么目标：** sampled reverse-KL、forward-KL、top-k、ranking、hidden-state 和混合 divergence 各有适用范围。
3. **如何分配 credit：** token-level 的局部概率变化不一定代表 action 对最终任务结果有帮助。
4. **如何避免副作用：** OPD 可能带来 OOD 遗忘、证据绕过、成功行为被侵蚀和长序列训练不稳定。

整体趋势是：**rollout policy、teacher context、support、divergence 和 credit assignment 必须联合设计，不能只替换 loss 函数。**

## 二、方法背景

### 2023-06-23：GKD/OPD 基础

**论文：** [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
**发表信息：** arXiv v1，后发表于 ICLR 2024

这项工作提出 Generalized Knowledge Distillation（GKD）：student 使用自身生成的序列训练，teacher 对这些序列提供反馈；训练目标不局限于传统 KL，也可以和 RL fine-tuning 结合。它解决了自回归模型 KD 中“训练看到的固定序列”和“推理时 student 自己生成的序列”之间的分布不一致问题，是当前 OPD 研究的基础。

### 2026-03-07：高熵位置的混合 KL

**论文：** [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079)
**发表信息：** arXiv v1；论文标注为 ICML 2026

论文指出 reverse-KL 的 mode-seeking 特性会损失生成多样性，并且在 teacher 高熵位置产生不稳定信号。Entropy-Aware OPD 在 teacher 高熵时加入 forward-KL，在低熵位置保留 reverse-KL。论文报告相对基线在 Qwen3-0.6B、1.7B、4B 上的 Pass@8 增益分别为 1.37、2.39、5.05。

## 三、按日期跟踪

以下按 arXiv 首次公告或最新版本日期倒序排列。同一天的论文按主题分组。

### 2026-11-05

#### Rollout 与 teacher 状态

**[CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation](https://arxiv.org/abs/2611.07959)**
用 maximal coupling 和 speculative decoding，把训练状态分布适度移向 teacher 可到达的状态，同时保留 OPD loss。论文在 Qwen3/Qwen3.5、0.6B-4B student 上报告相对 OPD 的 Pass@1 增益为 0.9-2.0，Pass@8 增益为 1.2-3.4。

**[O2PD: Overlap-Gated On-Policy Distillation on Reliable Prefix](https://arxiv.org/abs/2611.07731)**
发现 student rollout 后缀上的 teacher supervision 可能不可靠，因此只在 rollout 前 20% 的可靠 prefix 上应用监督，并通过 overlap-gated alpha-divergence、reset support 和 support convergence 处理 top-k 支持截断问题。

**[Learning Dynamics of On-Policy Distillation](https://arxiv.org/abs/2611.08016)**
将 OPD 的反馈回路拆成“distillation signal 转化为参数更新”和“参数更新改变后续 rollout”两阶段。论文强调，局部 teacher agreement 变好并不保证后续 rollout 变好；适当的 offline preparation 可能是 OPD 成功的前置条件。

#### 目标函数与信息来源

**[Reinforcing What the Student Has Learned: Teacher-Guided OPD](https://arxiv.org/abs/2611.07722)**
提出 Student-Delta OPD，用 student 与 base model 的 centered probability difference 作为更新系数，只保留方向上与 teacher-student feedback 一致的信号。论文在五种 teacher-student 配置上报告六任务数学平均提升 7.76-10.37 个百分点。代码：[aim-uofa/sd-opd](https://github.com/aim-uofa/sd-opd)。

### 2026-11-04

**[D2-OPD: Separating Direction and Distance in Extrapolative OPD](https://arxiv.org/abs/2611.07241)**
将 extrapolation 的方向与距离解耦：用 per-prefix hard KL 约束距离，用 teacher post-training shift 和 student-teacher gap 决定方向，再用 forward-KL 优化。三个设置中相对最强基线最高提升 9.45 个百分点。

**[EvoOPD: Evolving Privileged Information Strategies for Black-Box OPD](https://arxiv.org/abs/2611.06542)**
针对无法访问 logits 的 black-box teacher，把文本反馈视为 privileged information，并维护会随 student 状态变化的策略库。六个 model-task 设置均取得最佳整体结果，论文报告相对最强 baseline 平均提升：数学 2.14 个百分点，代码 1.30 个百分点。

### 2026-11-03

**[On the Generalization Limitation of OPD for Post-RL Consolidation](https://arxiv.org/abs/2611.05609)**
指出 IND 上接近 teacher 的分数可能掩盖 OOD 能力损失：在 extrapolative 设置中，OPD/MOPD 的 OOD 分数仍比 base model 低 2.45%-16.25%；compositional 设置的 OOD 表现比最佳 teacher 低 18.19%-24.34%。论文提出 geometry-based patching 和 base-preserving replay。

**[StepX-OPD](https://arxiv.org/abs/2611.04386)**
将 diffusion model 的多能力整合和 few-step distillation 合并为一个 OPD 过程，避免先训练 full-step teacher、再做 step distillation 的串行流程。论文在 SD3.5-Medium 和 FLUX.1-dev 上报告了更高的四步模型综合分数，并在 SD3.5-Medium 上减少约 37% GPU-hours。

### 2026-11-02

**[SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding](https://arxiv.org/abs/2611.02846)**
把 OPD rollout 从串行 draft-and-verify 改成 branch-parallel，并用 tree packing 共享 teacher prefix 计算，以 acceptance length 为直接优化目标。论文报告最高 41.45% 的 mean acceptance length 提升、16.09x supervised-token throughput 和 3.56x generation speedup。

**[Meet Students Where They Are: Student-Prefix Teacher Adaptation](https://arxiv.org/abs/2611.03133)**
先收集不同长度的 student prefix，再用 RLVR 训练 teacher 学会从这些错误或陌生状态继续完成任务，最后用适配后的 teacher 做标准 OPD。重点是修复 teacher-side distribution mismatch，而不是修改 OPD loss。

**[Replay for LLM Continual Learning with Multi-Teacher OPD](https://arxiv.org/abs/2611.02746)**
研究连续到达的新领域中如何避免 MOPD 的 catastrophic forgetting。每个阶段只把旧 prompt replay 给前一阶段 student，不保存所有旧 teacher 或 checkpoint；五领域实验显示，这种 replay 能接近保存全部 snapshot 的 recovery 效果。

### 2026-11-01

**[Order-Only On-Policy Distillation](https://arxiv.org/abs/2611.02112)**
提出 O3PD，只学习 teacher top-k token 的排序，不拟合 teacher 的概率值，因此对 temperature 不敏感。六个 teacher-student pair 均超过 OPD，平均最高提升 2.9 个百分点，AIME25 最高提升 5.6 个百分点。

**[Bernoulli On-Policy Distillation](https://arxiv.org/abs/2611.02093)**
指出 sampled reverse-KL 会系统性削弱高置信度 student token 的梯度。Bernoulli OPD 直接匹配该 token 的 teacher/student probability，以保留 disagreement 信号。代码：[duterscmy/Bernoulli-OPD](https://github.com/duterscmy/Bernoulli-OPD)。

**[Better Answers, Weaker Evidence Control](https://arxiv.org/abs/2611.01967)**
发现 OPD 提升原文档问答准确率的同时，可能让 student 忽略支持答案的证据变化。QASPER 实验中标准 OPD 使 counterfactual accuracy 下降 5.3 个百分点、evidence bypass 上升 15.7 个百分点；EP-OPD 恢复了其中 54%-66% 的损失。代码：[RuoranXu/EP-OPD](https://github.com/RuoranXu/EP-OPD)。

**[OC-OPD: Observation-coordinated Supervision](https://arxiv.org/abs/2611.01965)**
通过 student/teacher 作为 candidate source 和 evaluator 的四种 observation，协调 reverse-KL 与 forward-KL 的梯度方向，并按 distribution discrepancy 缩放更新。论文报告在数学推理、科学问答和代码生成上的跨任务收益。代码：[RuoranXu/OC-OPD](https://github.com/RuoranXu/OC-OPD)。

### 2026-10-30 至 2026-10-28

**[DecisionRank](https://arxiv.org/abs/2610.42128)**（10-30）将 item-space OPD 用于 decision-model reranker。1.7B student 超过 27B generative teacher 的 TREC DL 结果，8B 模型相对 generative listwise 版本最高快 24.5x。

**[Distilling the Shape of Uncertainty](https://arxiv.org/abs/2610.39020)**（10-28）提出 Entropy Shape Correlation，用“在哪里需要分支”补充 token-overlap 的“分支到什么 token”，并用于选择更适合的 teacher。

**[ATI-OPD](https://arxiv.org/abs/2610.38160)**（10-28）在长 reasoning prefix 出现高 disagreement 时，用短 teacher segment 替换最早的 divergence 区间，再让 student 继续生成，从而减少 gradient spike 和异常长输出。

**[Proposing Better Rollouts for OPD](https://arxiv.org/abs/2610.36477)**（10-27 更新）提出 MPOPD，用 student-anchored、teacher-guided geometric mixture 生成 rollout。Qwen3-4B-Base 数学任务平均准确率比 OPD 高 6.91 个百分点，同时平均生成长度降低 41%。

### 2026-10-25 至 2026-10-20

**[PhaseOPD](https://arxiv.org/abs/2610.34191)**（10-25）在 flow-matching 模型中根据生成 phase 决定何时 query teacher、何时提前 skip；SD3.5-M 上训练时间相对 DanceOPD 降低 41.3%。

**[Frontier-OPD](https://arxiv.org/abs/2610.32802)**（10-23）在共享状态 trajectory graph 中寻找“仍然困难但能自主续写”的 learning frontier，并过滤无效 teacher bridge。

**[OnQ](https://arxiv.org/abs/2610.29378)**（10-21）将 OPD 和 candidate-wise Dynamic KL 用于低比特 quantization recovery；2-bit Qwen3-8B 相对最强 baseline 的九任务平均提升为 10.26 个百分点。

**[On-Policy Transition Distillation](https://arxiv.org/abs/2610.28048)**（10-20）除最终层分布外，还对齐中间层之间的 log-probability transition；三域蒸馏约恢复 90% teacher-student gap，而普通 OPD 约为 75%。

**[Ask to Distill](https://arxiv.org/abs/2610.27450)**（10-20）不依赖答案标签，直接优化 synthetic question，使其在 student rollout 上激发更有价值的 teacher signal。代码：[GasolSun36/Ask_to_Distill](https://github.com/GasolSun36/Ask_to_Distill)。

**[Can Native Safety Travel?](https://arxiv.org/abs/2610.27555)**（10-20）研究把 safety pretraining 获得的 native safety 从较小 safety teacher 转移给普通 student，并支持跨 tokenizer transfer。

### 2026-10-19 至 2026-10-14

**[ToolOPD: Credit-Aware OPD for Tool-Use Agents](https://arxiv.org/abs/2610.25693)**（10-18）将完整工具轨迹和结果总结成 hindsight skill，再计算 skill 对每个 action token 的影响，形成 outcome-aware credit，并和 RL 联合优化。

**[OPDArena](https://arxiv.org/abs/2610.25651)**（10-18）系统比较 SFT、RLVR、OPD 的学习与遗忘，发现 teacher 的 post-training 来源会显著影响 capability retention。代码：[StellarCurtain/OPDArena](https://github.com/StellarCurtain/OPDArena)。

**[CD-OPD](https://arxiv.org/abs/2610.25359)**（10-18）用 teacher dependency graph 构造 contextual digest，并用 online meta-reweighting 降低 teacher takeover 和 collaboration failure。

**[SPOD](https://arxiv.org/abs/2610.25034)**（10-18）保护已经成功的 student decision，屏蔽与其冲突的 teacher supervision，并加入 soft probability floor。论文报告最终 success 最高提升 15.2 个百分点，代码：[Jike338/SPOD](https://github.com/Jike338/SPOD)。

**[GR-OPD](https://arxiv.org/abs/2610.23042)**（10-16）把 token feedback 聚合为 group-relative sequence credit，并保留受控的 collective teacher pressure；五种 Qwen3 设置均超过 vanilla OPD。

**[Divergence-Routed OPD](https://arxiv.org/abs/2610.19572)**（10-14）在高 JS divergence 位置额外加入 forward-KL，其他位置保留 reverse-KL；论文报告 severe teacher-mode under-coverage 相对 OPD 降低 37.9%。

**[Condition-Contrastive Extrapolation for OPD](https://arxiv.org/abs/2610.21195)**（10-14）用完整条件和 mask 条件的 teacher 分布做对比，减少违反题目 premise 的错误 continuation；invalid-continuation preference 从 27.0% 降到 16.0%。

## 四、核心判断

### 1. Rollout policy 和 loss 同等重要

student-only rollout 更接近部署，但会遗漏 teacher-reachable state。CoRD、MPOPD、Frontier-OPD 和 SPTA 分别从状态覆盖、混合 proposal、学习前沿和 teacher adaptation 解决这个问题。后续论文应同时报告 student rollout 和 teacher-reachable-state 上的指标。

### 2. Reverse-KL 是默认选项，不是最终答案

Reverse-KL 擅长 mode-seeking，但会低估 student 当前概率很低的 teacher-supported token，也可能在 teacher 高熵处损失多样性。当前更可靠的方向是按 divergence、entropy、overlap 或 prefix reliability 条件化地引入 forward-KL、ranking、bounded extrapolation 或 support correction。

### 3. Token-level supervision 不等于正确 credit assignment

长链路 agent 中，一个 action 的局部 log-probability gain 不一定提升最终结果。因此 ToolOPD、GR-OPD、SPOD 以及 error-truncated 方法开始引入 outcome、group、success 和 error boundary 信息。

### 4. IND 分数不能单独证明能力转移

应同时检查 OOD/compositional generalization、counterfactual evidence、旧能力 retention、termination、response length、repetition 和 student-only inference。否则可能把 evidence bypass 或 OOD forgetting 误判为成功蒸馏。

### 5. Teacher context 是算法的一部分

teacher 在 clean prompt 上很强，不代表它能可靠评价错误 student prefix。teacher adaptation、privileged-information strategy、condition contrast 和 takeover-informed repair 都说明 teacher 的输入状态必须显式设计和评估。

## 五、推荐阅读顺序

1. [GKD/OPD 基础论文](https://arxiv.org/abs/2306.13649)：理解 student-generated trajectory 和分布偏移。
2. [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079)：理解 forward/reverse-KL 的基本权衡。
3. [CoRD](https://arxiv.org/abs/2611.07959)、[MPOPD](https://arxiv.org/abs/2610.36477)、[SPTA](https://arxiv.org/abs/2611.03133)：理解 rollout 和 teacher-state mismatch。
4. [DrOPD](https://arxiv.org/abs/2610.19572)、[O3PD](https://arxiv.org/abs/2611.02112)、[D2-OPD](https://arxiv.org/abs/2611.07241)：理解 objective、support 和 extrapolation。
5. [ToolOPD](https://arxiv.org/abs/2610.25693)、[GR-OPD](https://arxiv.org/abs/2610.23042)、[SPOD](https://arxiv.org/abs/2610.25034)：理解 agentic credit assignment 和稳定性。
6. [Learning Dynamics](https://arxiv.org/abs/2611.08016)、[OPD Generalization Limitation](https://arxiv.org/abs/2611.05609)：在解读 benchmark 前了解反馈回路和 OOD 风险。

## 六、建议的复现实验记录

每次新增方法至少记录：

- teacher、student、tokenizer、初始化和参数规模；
- prompt 集、rollout 数、sampling temperature、最大长度和停止规则；
- optimizer steps、生成 token 数、teacher forward 次数和 GPU-hours；
- sampled/top-k/full-vocabulary support、teacher temperature 和 divergence；
- teacher refresh、checkpoint 选择和 early stopping 规则；
- accuracy/pass@k、长度、termination、entropy、calibration、repetition；
- OOD/compositional、counterfactual evidence 和旧能力 retention；
- rollout policy、teacher context、support、credit weighting 的 ablation。

## 七、来源与更新规则

本报告的论文元数据和摘要于 **2026-11-06** 通过 arXiv 搜索/API 页面核对；代码链接仅在论文或 arXiv 记录明确提供时收录。报告中的“最新”指截至该日期在 arXiv 已公告或更新的工作，不等同于同行评审接收或生产可用。

下一次更新时，按日期追加新的研究记录，并核对：

1. arXiv ID、版本号和首次公告日期；
2. 论文摘要中的方法与实验数字；
3. 代码仓库、commit/release 状态；
4. 数字比较的 baseline 是否确实为 vanilla OPD；
5. 是否存在 OOD、长度、停止和能力 retention 等副作用指标。
