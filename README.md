# OPD Papers

> On-Policy Distillation for language models: papers, implementations, and research notes.

Research snapshot: **2026-11-06**. The papers below are arXiv preprints unless a venue is explicitly stated. Results are reported as claimed by the papers and should be reproduced before being treated as definitive.

## Executive Summary

On-policy distillation (OPD) trains a student on prefixes and responses sampled from the student's own policy, while a stronger teacher supplies token-level distributions or scores. Compared with ordinary/off-policy KD, OPD reduces train/inference distribution mismatch and can transfer reasoning behavior without teacher-written trajectories. Its central weakness is that the teacher is queried on states the teacher may not naturally visit, while the student's rollout distribution changes after every update.

The latest work is converging on five directions:

1. **Fix the rollout distribution.** Standard student-only rollouts miss teacher-reachable states. CoRD uses maximal coupling to move only the training-state distribution toward the teacher; MPOPD uses a student-anchored geometric mixture; SPTA adapts the teacher to student prefixes.
2. **Make the objective selective rather than blindly reverse-KL.** DrOPD routes forward-KL corrections to high-divergence positions; O2PD gates learning to reliable prefixes and repairs top-k support; O3PD transfers token order without copying teacher calibration; Bernoulli OPD restores signal for high-confidence sampled tokens.
3. **Assign credit at the right level.** ToolOPD, GR-OPD, and related agentic methods move beyond uniform token credit by using outcome-aware, group-relative, or error-boundary-aware weighting.
4. **Preserve useful student behavior and generalization.** SPOD prevents erosion of successful student choices; EP-OPD protects evidence responsiveness; recent analyses show that in-distribution matching can hide out-of-distribution forgetting.
5. **Reduce the cost and broaden the scope.** SpecOPD parallelizes speculative-decoding rollouts, while OPD variants are now being applied to tool agents, retrieval, recommendation, diffusion/video/audio generation, quantization recovery, safety transfer, and multi-teacher consolidation.

The field's main unresolved question is no longer whether on-policy data matters. It is **which student states should be supervised, by which teacher view, with which divergence and credit assignment, and how to avoid reinforcing the student's own errors**.

## Core Method

Let `x` be a prompt, `y ~ pi_theta(. | x)` a student rollout, and `p_T(. | x, y_<t)` and `p_S(. | x, y_<t)` the teacher and student next-token distributions on the student's prefix. A common sampled reverse-KL OPD objective is equivalent to updating the sampled token with a teacher-student log-probability difference:

```text
L_OPD = - sum_t log p_S(y_t | x, y_<t)
                     * stopgrad(log p_T(y_t | x, y_<t)
                                - log p_S(y_t | x, y_<t))
```

Implementations differ in whether they use the sampled token, teacher top-k support, full-vocabulary KL, forward/reverse mixtures, ranking losses, hidden-state signals, or outcome-level weights. The design axes that should always be reported are:

- rollout policy and sampling temperature;
- teacher context and any privileged information;
- support truncation and tokenizer alignment;
- divergence and gradient weighting;
- stopping, truncation, and long-response handling;
- task verifier, reward, or outcome labels;
- teacher/student model pair and initialization;
- online refresh schedule and compute budget.

## Research Timeline

### Foundations

| Year | Work | Main contribution |
| --- | --- | --- |
| 2023/2024 | [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) | Introduces Generalized Knowledge Distillation (GKD), training on student-generated sequences and allowing alternative divergences and RL integration. This is the practical ancestor of the current OPD literature. |
| 2026 | [Entropy-Aware On-Policy Distillation](https://arxiv.org/abs/2603.07079) | Adds forward-KL behavior when teacher entropy is high, preserving diversity while retaining reverse-KL precision. Reported Pass@8 gains over OPD: +1.37, +2.39, and +5.05 for Qwen3-0.6B, 1.7B, and 4B students; accepted at ICML 2026. |

### October-November 2026: Latest Progress

#### Rollout distribution and teacher-state mismatch

| Date | Paper | What changed | Reported result |
| --- | --- | --- | --- |
| 2026-11-05 | [CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation](https://arxiv.org/abs/2611.07959) | Uses maximal coupling/speculative decoding to interpolate toward teacher-reachable states while retaining the OPD loss. | +0.9 to +2.0 Pass@1 and +1.2 to +3.4 Pass@8 over OPD across Qwen3/Qwen3.5 students. |
| 2026-11-02 | [Meet Students Where They Are](https://arxiv.org/abs/2611.03133) | RLVR-adapts the teacher on random-length student prefixes before standard OPD. | Improves over standard OPD and task-level RLVR-adapted teachers on math benchmarks. |
| 2026-10-28 | [Proposing Better Rollouts for On-Policy Distillation](https://arxiv.org/abs/2610.36477) | MPOPD uses a student-anchored, teacher-guided geometric-mixture proposal policy. | +6.91 average math accuracy over OPD for Qwen3-4B-Base and 41% shorter generation. |
| 2026-10-23 | [Frontier-OPD](https://arxiv.org/abs/2610.32802) | Selects guidance states on a shared-state trajectory graph and filters teacher bridges by subsequent student continuation. | Higher autonomous performance on ALFWorld, WebShop, and ScienceWorld. |

#### Objective, support, and credit assignment

| Date | Paper | What changed | Reported result |
| --- | --- | --- | --- |
| 2026-11-05 | [Student-Delta OPD](https://arxiv.org/abs/2611.07722) | Uses centered student-base deltas as coefficients, retaining only signs consistent with teacher-student feedback. | +7.76 to +10.37 six-task math average points over OPD across five configurations. Code: [aim-uofa/sd-opd](https://github.com/aim-uofa/sd-opd). |
| 2026-11-05 | [O2PD: Overlap-Gated OPD](https://arxiv.org/abs/2611.07731) | Applies supervision on a reliable rollout prefix, then uses overlap-gated alpha-divergence with reset/support-convergence strategies instead of fixed teacher top-k support. | Addresses unreliable off-policy suffixes, discarded tail mass, and dominant-mode mismatch; quantitative comparisons are reported in the preprint. |
| 2026-11-04 | [D2-OPD](https://arxiv.org/abs/2611.07241) | Separates extrapolation direction from distance using a hard KL constraint and forward-KL optimization. | Up to +9.45 points over the strongest baseline in three settings. |
| 2026-11-04 | [EvoOPD](https://arxiv.org/abs/2611.06542) | Evolves a bank of privileged-information strategies for black-box teachers according to their current repair utility. | Best overall in six model-task settings; +2.14 math and +1.30 code points over the strongest baselines on average. |
| 2026-11-01 | [Order-Only OPD](https://arxiv.org/abs/2611.02112) | Matches teacher top-k token ordering instead of temperature-sensitive probabilities. | Beats OPD in all six teacher-student pairs, up to +2.9 average points and +5.6 on AIME25. |
| 2026-11-01 | [Bernoulli OPD](https://arxiv.org/abs/2611.02093) | Directly matches teacher/student probabilities for the sampled token to avoid confidence-dependent gradient suppression. | Restores high-confidence-token signal; code: [duterscmy/Bernoulli-OPD](https://github.com/duterscmy/Bernoulli-OPD). |
| 2026-10-14 | [Divergence-Routed OPD](https://arxiv.org/abs/2610.19572) | Adds forward KL only at high Jensen-Shannon-divergence positions while retaining reverse KL elsewhere. | Reduces severe teacher-mode under-coverage by 37.9% relative to OPD. |
| 2026-10-14 | [Condition-Contrastive Extrapolation for OPD](https://arxiv.org/abs/2610.21195) | Contrasts teacher predictions with a condition-masked prompt to suppress continuations that violate problem premises. | Invalid-continuation preference falls from 27.0% to 16.0%; +1.60/+1.95 avg@8 over OPD for 0.6B/1.7B students. |
| 2026-10-19 | [ToolOPD](https://arxiv.org/abs/2610.25693) | Converts completed tool-use outcomes into hindsight skills and action-level credit, jointly trained with RL. | Reports gains over RL and self-distillation baselines on long-horizon tool-use tasks. |
| 2026-10-16 | [GR-OPD](https://arxiv.org/abs/2610.23042) | Aggregates token feedback into group-relative sequence credit, with a controlled collective teacher pressure term. | Outperforms vanilla OPD in all five Qwen3 settings; remains stable under mismatched thinking patterns. |

#### Reliability, retention, and generalization

| Date | Paper | Finding or intervention |
| --- | --- | --- |
| 2026-11-05 | [Learning Dynamics of OPD](https://arxiv.org/abs/2611.08016) | Separates signal-to-update conversion from update-to-rollout broadcast; explains why local teacher agreement can still worsen future rollouts. Offline preparation can make later OPD effective. |
| 2026-11-03 | [Generalization Limitation of OPD for Post-RL Consolidation](https://arxiv.org/abs/2611.05609) | IND recovery can coexist with 2.45%-16.25% OOD loss versus the base model; geometry-based patching and base-preserving replay are proposed. |
| 2026-11-01 | [Evidence Bypass in OPD](https://arxiv.org/abs/2611.01967) | Standard OPD raises original-document accuracy but can weaken response to changed evidence. EP-OPD recovers 54%-66% of counterfactual-accuracy loss while preserving 80%-85% of the original gain. Code: [RuoranXu/EP-OPD](https://github.com/RuoranXu/EP-OPD). |
| 2026-10-18 | [OPDArena](https://arxiv.org/abs/2610.25651) | Compares SFT, RLVR, and OPD across learning and forgetting; teacher post-training source strongly affects retention. Code: [StellarCurtain/OPDArena](https://github.com/StellarCurtain/OPDArena). |
| 2026-10-18 | [CD-OPD](https://arxiv.org/abs/2610.25359) | Uses a teacher dependency-graph digest plus online meta-reweighting to reduce teacher takeover/collaboration failure. |
| 2026-10-18 | [SPOD](https://arxiv.org/abs/2610.25034) | Masks teacher updates that erode successful student decisions and adds a probability floor. | Up to +15.2 final success points and up to +20.0 over teacher on agentic benchmarks. Code: [Jike338/SPOD](https://github.com/Jike338/SPOD). |

#### Efficiency and new applications

| Date | Paper | Application or systems contribution |
| --- | --- | --- |
| 2026-11-02 | [SpecOPD](https://arxiv.org/abs/2611.02846) | Branch-parallel OPD for speculative decoding with tree packing and an acceptance-length objective. | Up to 41.45% higher mean acceptance length, 16.09x supervised-token throughput, and 3.56x generation speedup. |
| 2026-10-30 | [DecisionRank](https://arxiv.org/abs/2610.42128) | Item-space OPD transfers a generative ranker's knowledge to a fast score-based reranker. | 1.7B student surpasses a 27B teacher on TREC DL; up to 24.5x faster. |
| 2026-10-27 | [OnQ](https://arxiv.org/abs/2610.29378) | Combines OPD with candidate-wise dynamic KL for low-bit quantization recovery. | At 2-bit Qwen3-8B, +10.26 average points over the strongest baseline. |
| 2026-10-20 | [On-Policy Transition Distillation](https://arxiv.org/abs/2610.28048) | Aligns layer-to-layer log-probability transitions in addition to final distributions. | Recovers about 90% of the multi-domain teacher gap versus about 75% for OPD. |
| 2026-10-20 | [Can Native Safety Travel?](https://arxiv.org/abs/2610.27555) | Transfers safety-pretraining behavior, including across tokenizers, with a smaller safety teacher. |
| 2026-10-20 | [Ask to Distill](https://arxiv.org/abs/2610.27450) | Optimizes synthetic questions for the teacher signal they elicit, without answer labels. Code: [GasolSun36/Ask_to_Distill](https://github.com/GasolSun36/Ask_to_Distill). |
| 2026-10-20 | [On-Policy Distillation for Autonomous Vehicle Motion Planning](https://arxiv.org/abs/2604.07944) | Applies GKD/OPD to waypoint planning; the student approaches teacher performance at 5x lower model size. |

## What the Evidence Suggests

### 1. Rollout design is as important as the loss

Student-only rollouts are attractive because they match deployment, but they create a blind spot: teacher-supported states that the student never reaches receive no supervision. CoRD, MPOPD, Frontier-OPD, and SPTA address this from different sides. The practical trade-off is between deployment realism and coverage of useful teacher states. Any new OPD method should report both student-rollout and teacher-reachable-state evaluations.

### 2. Reverse KL is a useful default, not a universal answer

Reverse KL is mode-seeking and gives strong local imitation, but it under-covers teacher modes that the student currently considers unlikely and can suppress diversity at high teacher entropy. The strongest recent pattern is conditional hybridization: retain reverse KL where the teacher is reliable and use forward KL, ranking, bounded extrapolation, or a support correction only where diagnostics justify it.

### 3. Token-level density does not solve credit assignment

Dense supervision can still be wrong in long-horizon tasks. A token's local probability shift may not reflect whether the action helped the eventual outcome. ToolOPD, GR-OPD, SPOD, and error-truncated methods show that outcome, group, and success information should influence where and how strongly distillation is applied.

### 4. Better in-distribution scores can hide regressions

OPD can increase accuracy on the training-like distribution while reducing evidence responsiveness, retaining less OOD capability, or eroding successful student behaviors. Evaluation should therefore include counterfactual evidence tests, OOD/compositional tasks, capability retention, response length, termination rate, and independent student-only inference.

### 5. The teacher is a state-dependent instrument

A teacher that is strong on clean prompts may be unreliable on erroneous student prefixes. Teacher adaptation, privileged-information design, condition contrast, and takeover-informed repair all treat the teacher's queried context as part of the algorithm rather than a fixed implementation detail.

## Practical Reading Order

1. Read [GKD/OPD (2023)](https://arxiv.org/abs/2306.13649) for the core train-on-student-rollouts idea.
2. Read [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) for the reverse-KL versus forward-KL trade-off.
3. Read [CoRD](https://arxiv.org/abs/2611.07959), [MPOPD](https://arxiv.org/abs/2610.36477), and [SPTA](https://arxiv.org/abs/2611.03133) for rollout/teacher mismatch.
4. Read [DrOPD](https://arxiv.org/abs/2610.19572), [Order-Only OPD](https://arxiv.org/abs/2611.02112), and [D2-OPD](https://arxiv.org/abs/2611.07241) for objective design.
5. Read [ToolOPD](https://arxiv.org/abs/2610.25693), [GR-OPD](https://arxiv.org/abs/2610.23042), and [SPOD](https://arxiv.org/abs/2610.25034) for credit assignment and agentic stability.
6. Read [Learning Dynamics](https://arxiv.org/abs/2611.08016) and [OPD Generalization Limitation](https://arxiv.org/abs/2611.05609) before interpreting benchmark gains.

## Suggested Benchmark Protocol

For a fair OPD comparison, keep the following fixed and disclose them:

- teacher, student, tokenizer, initialization, and parameter count;
- prompt set, rollout count, sampling temperature, and maximum length;
- number of optimizer steps, generated tokens, teacher forward passes, and GPU-hours;
- loss support (`sampled`, `top-k`, or `full vocabulary`) and teacher temperature;
- online teacher refresh and checkpoint selection rules;
- exact-match/pass@k plus length, termination, entropy, calibration, and repetition;
- OOD/compositional tasks, evidence counterfactuals, and capability-retention tests;
- ablations for rollout policy, divergence, support, credit weighting, and teacher context.

## Sources and Update Policy

The paper metadata and abstracts in this snapshot were checked against arXiv search/API pages on 2026-11-06. Repository links are included only when a paper or its arXiv record explicitly provided them. “Latest” means newly announced or updated on arXiv by the snapshot date, not peer-review acceptance or production readiness.

To update this repository, search arXiv for `"on-policy distillation"`, `OPD`, and method-specific names, record the arXiv identifier and announcement date, then verify the abstract, version, code URL, and whether reported numbers are against vanilla OPD or another baseline.
