# CTDreamer 开发指南

**最后核对：** 2026-08-17
**代码基准：** `main`，提交 `546e4fa`，外加当前工作区的 Phase 2 实现
**当前状态：** Phase 2 核心已实现；真实 `delta_t` 数据管线和完整连续时间实验尚未完成。

## 项目定位

CTDreamer 基于 R2-Dreamer 开发连续时间世界模型。项目不面向电力系统，优先在常用强化学习环境中验证 Neural ODE 相比离散时间 RSSM 的优势。

拟议方法暂称 **CT-R2Dreamer（Continuous-Time R2-Dreamer）**。核心目标是：

1. 保留 R2-Dreamer 的单帧 encoder、categorical stochastic state、prior/posterior、R2 projector 和 actor-critic；
2. 用 action-conditioned Neural ODE 替换 RSSM 中的 block-GRU deterministic transition；
3. 显式记录并使用物理时间间隔 `delta_t`；
4. 在固定步长、非规则步长和未见步长上公平比较 GRU、`delta_t`-GRU 与 Neural ODE；
5. 第一阶段不使用 temporal encoder、不让 actor 选择动作持续时间、不使用 `odeint_event`。

本项目主要修改当前仓库；`dtak/mbrl-smdp-ode` 只作为 SMDP、动作持续时间和连续时间建模的参考，不直接移植旧训练框架。

## 当前代码基线

当前主链路为：

```text
trainer / replay buffer
        │
        ▼
Dreamer.update
        │
        ├── MultiEncoder
        ├── RSSM.observe / prior / imagine_with_action
        ├── reward + continue heads
        ├── R2 projector / alternative representation objective
        └── imagined actor-critic + replay value learning
```

当前 `rssm.py` 中：

- `Deter` 是 block-GRU 风格的 deterministic transition；
- `RSSM._img_net` 是 categorical prior；
- `RSSM._obs_net` 是 observation-conditioned posterior；
- `stoch` 的形状为 `(B, S, K)`，`deter` 的形状为 `(B, D)`；
- `get_feat` 返回 flattened stochastic state 与 deterministic state 的拼接；
- `dyn_loss` 使用 $D_{\mathrm{KL}}(\operatorname{sg}(q)\,\|\,p)$；
- `rep_loss` 使用 $D_{\mathrm{KL}}(q\,\|\,\operatorname{sg}(p))$。

当前 `dreamer.py` 中：

- `MultiEncoder` 同时支持 CNN 和 MLP observation；
- `rep_loss=r2dreamer` 时不构建 decoder，而是构建线性 `Projector`；
- R2 loss 将 projected posterior feature 与 detached encoder embedding 做 Barlow Twins 对齐；
- reward 使用 symlog two-hot distribution；
- continue 使用 Bernoulli distribution；
- actor、critic、slow critic 和 replay value learning 已存在；
- imagination 使用冻结的 world model 和 actor 副本；
- 当前所有损失由一个 optimizer 统一更新，但梯度路由由 detach/frozen module 控制。

不要在没有对照实验的情况下重写这些无关模块。

## 时间与数据契约

连续时间实验的基本 transition 必须包含：

$$
\mathcal D_k =
\left(o_k, a_k, \Delta t_k, R_k, c_k, o_{k+1}\right).
$$

定义：

- $t_k$：当前决策/观测时刻；
- $\Delta t_k=t_{k+1}-t_k$；
- $a_k$：在区间 $[t_k,t_{k+1})$ 内保持不变的动作；
- $R_k$：该区间内累计 reward，而不是与持续时间无关的抽象单步 reward；
- $c_k$：到达 $t_{k+1}$ 后是否继续；
- episode 的比较必须保持相同物理时间范围或相同底层 simulator step 预算。

`delta_t` 应保存为真实物理时间或相对于基础控制步长的明确倍数。不要只保存 action-repeat index 而丢失单位。

## 拟议混合状态

定义 latent state：

$$
s_k=(h_k,z_k).
$$

- $h(t)\in\mathbb R^D$：连续、确定性状态；
- $z_k\in\{1,\ldots,K\}^S$：边界处更新的离散随机状态；
- observation embedding 记为 $e_k$；
- $h_k$ 与 $z_k$ 共同作为 actor、critic 和预测 head 的 latent feature。

区间内部只积分 $h(t)$；$z_k$ 与 $a_k$ 在区间内保持常值。到达已知边界 $t_{k+1}$ 后，prior 或 posterior 再更新 $z_{k+1}$。

不要把 categorical `z` 当成普通连续变量交给 ODE solver，也不要在第一版使用 continuous relaxation 统一积分 `h` 与 `z`。

## 模块设计

### Image Encoder

第一版保留当前 `MultiEncoder`：

$$
e_k=E_\phi(o_k).
$$

- 图像环境继续使用单帧 `ConvEncoder`；
- DMC Proprio 继续使用 MLP encoder；
- 不增加 temporal encoder；
- 不给 encoder 输入绝对时间；
- temporal memory 由 `(h_k, z_k)` 提供。

理由：保持 encoder 一致，才能把变化归因到 transition。Temporal encoder 只作为稀疏像素观测的后续独立消融，并且必须同时提供给离散和 ODE 基线。

### Sequence Model：Continuous Deter

第一版用 `ContinuousDeter` 替换当前 `Deter`。连续模型直接复用原 block-GRU 的三路输入投影、RMSNorm、SiLU、block grouping、`BlockLinear`、reset/candidate/update gate，不再引入额外的 rate 映射、可学习衰减项或独立 MLP 向量场。

原 `Deter` 的离散更新为：

$$
\begin{aligned}
r_k
&=\sigma(r_k^{\mathrm{raw}}),\\
\widetilde h_k
&=\tanh\!\left(r_k\odot c_k^{\mathrm{raw}}\right),\\
u_k
&=\sigma\!\left(u_k^{\mathrm{raw}}-1\right),\\
h_{k+1}
&=(1-u_k)\odot h_k+u_k\odot\widetilde h_k.
\end{aligned}
$$

在每次 vector-field evaluation 中，根据当前 $h(t)$ 以及区间内保持不变的 $z_k,a_k$，使用与原 `Deter` 相同的 block core 计算：

$$
\left(
r_\theta^{\mathrm{raw}},
c_\theta^{\mathrm{raw}},
u_\theta^{\mathrm{raw}}
\right)
=
\operatorname{BlockCore}_\theta
\!\left(h(t),z_k,a_k\right).
$$

候选状态和 update gate 与原 `Deter` 完全相同：

$$
\begin{aligned}
r_\theta(t)
&=\sigma\!\left(r_\theta^{\mathrm{raw}}(t)\right),\\
\widetilde h_\theta(t)
&=\tanh\!\left(
r_\theta(t)\odot c_\theta^{\mathrm{raw}}(t)
\right),\\
u_\theta(t)
&=\sigma\!\left(u_\theta^{\mathrm{raw}}(t)-1\right).
\end{aligned}
$$

连续速度直接定义为原离散更新相对当前状态的位移：

$$
\boxed{
\frac{\mathrm d h(t)}{\mathrm dt}
=
u_\theta(t)\odot
\left[\widetilde h_\theta(t)-h(t)\right]
}
$$

训练时采用简单 CFO 方案，不调用 ODE solver，而使用线性区间 Euler：

$$
h_{k+1}
=
h_k+\Delta t_k f_\theta(h_k,z_k,a_k).
$$

当 $\Delta t_k=1$ 时：

$$
\begin{aligned}
h_{k+1}
&=h_k+u_k\odot(\widetilde h_k-h_k)\\
&=(1-u_k)\odot h_k+u_k\odot\widetilde h_k,
\end{aligned}
$$

因此单位时长线性 transition 与原 `Deter` 数值完全一致。推理、在线 `act`、prior rollout 和 actor imagination 才通过 torchode 积分：

$$
h_{k+1}
=
\operatorname{ODESolve}
\!\left(
f_\theta,
h_k,
z_k,
a_k,
\Delta t_k
\right).
$$

简单 CFO 速度监督使用现有无偏置 projector：

$$
\widehat v_k^e
=P\!\left([0,f_\theta(h_k,z_k,a_k)]\right),
\qquad
v_k^{e,*}
=\operatorname{stopgrad}\!\left(
\frac{e_{k+1}-e_k}{\Delta t_k}
\right),
$$

$$
L_{\mathrm{velocity}}
=
\operatorname{MSE}(\widehat v_k^e,v_k^{e,*}).
$$

跨 episode reset 和零时长区间不参与速度损失。训练目标只在观测边界使用线性区间速度，不采样区间内部 $\tau$，不使用 spline、FIR 或 bridge noise。

实现约束：

- 保留当前三路输入投影、动作幅值归一化、`BlockLinear`/8-block grouping、reset/candidate/update 输出以及 update logit 的 `-1` 偏置；
- action 在一次积分区间内使用 zero-order hold；
- 默认使用 autonomous vector field，不输入绝对 `t`；
- ODE 使用 $\Delta t_k$ 积分；replay 同时保留原始 $\Delta t_k$，供 reward、continue、discount 和指标使用；
- solver step size 和 tolerance 必须由配置显式给出；
- episode reset 时重置 `h`、`z` 和 previous action；
- `observe`、`obs_step`、`img_step`、`imagine_with_action` 均显式接收 `delta_t`；
- 每次实验记录 solver、step size/tolerance、NFE 和 $u_\theta$ 的统计量。

当前求解器设计：

- 使用 `torchode==1.0.1`；
- 默认 `AutoDiffAdjoint + Tsit5 + IntegralController`，直接对求解器操作做自动微分；
- `backprop_through_step_size_control=False`，不对自适应步长控制本身反传；
- `ode_step_size` 是 Tsit5 的归一化初始步长，也是 Heun/Euler 消融的固定步长；
- `ode_max_steps`、`rtol` 和 `atol` 由配置显式给出；
- ODE 子模块使用非 `fullgraph` 的局部 `torch.compile`；solver 的数据依赖循环不能整图捕获；
- fixed-step Heun/Euler 保留为效率与求解器消融。

选择 `AutoDiffAdjoint` 的依据是 size12M ODE 核心上的实测结果。torchode 的 batch-independent `BacksolveAdjoint` 为每个样本维护参数伴随状态，显存近似按 $B\lvert\theta\rvert$ 增长：batch 16 的 Tsit5 反向约使用 6.37 GiB，而 AutoDiff 约 61 MiB；batch 32 已达到约 12.7 GiB。Backsolve 只作为小模型、小 batch 消融，不作为训练默认值。

不要使用 `odeint_event`。当前 transition 边界由 `delta_t` 已知，不需要求事件根。只有研究区间内部未知时刻的碰撞、模式切换或终止时，才重新评估 event solver。

该设计的优点是结构最小、单位时长下与原 `Deter` 精确对应，并能处理任意 duration。代价是 update gate 被直接解释为单位时间速度系数，不再具有严格的解析指数时间常数；ODE 推理与线性训练之间也存在离散化差异，必须通过 solver 和 fixed-duration 消融验证。

### Dynamics Predictor

Dynamics Predictor 是边界处的 stochastic prior：

$$
p_\theta\!\left(z_{k+1}\mid h_{k+1}\right).
$$

- 保留 factorized categorical distribution；
- 保留 unimix 和 straight-through one-hot sample；
- observed rollout 与 imagined rollout 共用同一个 prior；
- imagined rollout 中只能使用 prior，不得访问未来 observation embedding。

Sequence Model 描述 deterministic continuous flow；Dynamics Predictor 描述随机性与离散模式。不要把二者合并成一个含义不清的“大 transition network”。

### Representation Model

Representation Model 是 observation-conditioned posterior：

$$
q_\phi\!\left(z_{k+1}\mid h_{k+1},e_{k+1}\right).
$$

- 真实交互和 posterior rollout 使用 `q`；
- imagination 使用 `p`；
- 第一版 observation 只通过 $z_{k+1}$ 校正状态；
- 不直接对 $h_{k+1}$ 做额外 GRU jump。

ODE-RNN 式 deterministic jump 是后续可选消融，不是默认方法。加入 jump 后必须区分 `h_minus` 和 `h_plus`，并承认 deterministic state 不再全程连续。

### Reward Predictor

第一版预测整个 SMDP 区间的累计 reward：

$$
p_\theta\!\left(
R_k\mid s_k
\right).
$$

默认保留 symlog two-hot distribution 和 negative log-likelihood。

连续 reward-rate 扩展：

$$
\widehat R_k
=
\int_{t_k}^{t_{k+1}}
r_\theta\!\left(h(t),z_k\right)\,\mathrm dt.
$$

该扩展只作为独立消融，不与第一版 ODE replacement 同时引入。

### Continue Predictor

第一版使用 duration-conditioned Bernoulli：

$$
p_\theta\!\left(
c_k\mid s_k
\right).
$$

Continue Predictor 使用区间起点、终点和 duration。训练时这里的 $s_{k+1}$ 必须来自 Dynamics Predictor 的 prior rollout，而不是依赖 $e_{k+1}$ 的 posterior correction；否则训练阶段会获得 imagination 中不存在的未来观测信息。

### Projector

默认保留当前线性无 bias `Projector`：

$$
x_k=P_\psi\!\left([h_k,z_k]\right),
\qquad
y_k=\operatorname{sg}(e_k).
$$

Projector 的职责仅是 R2 representation regularization：

- 不作为 actor 输入；
- 不作为 critic 输入；
- 不参与 ODE vector field；
- 不参与环境推理状态；
- 不要把 projected feature 当作新的 latent state。

第一版只对 posterior feature 使用 R2 loss。Prior-to-future-embedding predictive projector 可作为后续消融，但不能在核心 ODE 实验中默认加入。

## 世界模型损失

### Dynamics KL

$$
\mathcal L_{\mathrm{dyn}}
=
\max\!\left(
\tau_{\mathrm{free}},
D_{\mathrm{KL}}
\left[
\operatorname{sg}(q(z_k))
\,\|\,p(z_k)
\right]
\right).
$$

主要训练 continuous sequence model 和 prior。

### Representation KL

$$
\mathcal L_{\mathrm{rep}}
=
\max\!\left(
\tau_{\mathrm{free}},
D_{\mathrm{KL}}
\left[
q(z_k)
\,\|\,\operatorname{sg}(p(z_k))
\right]
\right).
$$

约束 posterior 与 dynamics prior 保持一致。

### R2 loss

对 projected latent $x$ 和 detached encoder embedding $y$ 做 batch-time 标准化，计算 cross-correlation $C$：

$$
C_{ij}
=
\frac{1}{N}
\sum_{n=1}^{N}
\bar x_{n,i}\bar y_{n,j},
$$

$$
\mathcal L_{\mathrm{R2}}
=
\sum_i(1-C_{ii})^2
+
\lambda_{\mathrm{off}}
\sum_{i\ne j}C_{ij}^2.
$$

### Prediction loss

$$
\mathcal L_{\mathrm{reward}}
=
-\mathbb E_{\mathcal D}
\log p_\theta
\!\left(
R_k\mid s_k,a_k,\Delta t_k
\right),
\qquad
\mathcal L_{\mathrm{continue}}
=
-\mathbb E_{\mathcal D}
\log p_\theta
\!\left(
c_k\mid s_k,a_k,s_{k+1},\Delta t_k
\right).
$$

### 总世界模型目标

$$
\begin{aligned}
\mathcal L_{\mathrm{WM}}
={}&
\beta_{\mathrm{dyn}}\mathcal L_{\mathrm{dyn}}
+
\beta_{\mathrm{rep}}\mathcal L_{\mathrm{rep}}
+
\beta_{\mathrm{R2}}\mathcal L_{\mathrm{R2}}
\\
&+
\beta_{\mathrm{reward}}\mathcal L_{\mathrm{reward}}
+
\beta_{\mathrm{continue}}\mathcal L_{\mathrm{continue}}.
\end{aligned}
$$

第一版不默认加入 vector-field norm、Jacobian penalty、semigroup penalty 或 latent reconstruction loss。先以结构稳定性和基础 loss 验证方法，再逐项消融。

## 连续时间 imagination 与折扣

第一版 `delta_t` 是外生变量，不由 actor 选择。Imagination 可以从 replay 的 duration 分布采样，或使用实验预先指定的 duration schedule。

必须使用时间一致的 discount：

$$
\gamma_k
=
\gamma_{\mathrm{base}}^{\Delta t_k/\Delta t_{\mathrm{base}}}
=
\exp(-\kappa\Delta t_k),
\qquad
d_k=\widehat c_k\gamma_k.
$$

权重：

$$
w_0=1,
\qquad
w_k=\prod_{i<k}d_i.
$$

时间感知 $\lambda$-return：

$$
G_k^\lambda
=
\widehat R_k
+
d_k
\left[
(1-\lambda)V(s_{k+1})
+
\lambda G_{k+1}^{\lambda}
\right].
$$

不要在 variable-`delta_t` 实验中继续使用与 duration 无关的固定 transition discount。Imagination horizon 应优先按物理时间定义，并通过 mask 处理 batch 内不同累计时间。

## Actor loss

保持当前 likelihood-ratio 风格：

$$
A_k
=
\frac{G_k^\lambda-V(s_k)}
{\operatorname{ReturnScale}},
$$

$$
\mathcal L_{\mathrm{actor}}
=
-\mathbb E
\left[
w_k
\left(
\log\pi_\eta(a_k\mid s_k)\operatorname{sg}(A_k)
+
\alpha_{\mathrm H}
\mathcal H[\pi_\eta(\cdot\mid s_k)]
\right)
\right].
$$

梯度约束：

- actor 输入原始 `concat(h, z)`；
- actor loss 只更新 actor；
- world model 在 actor imagination 中冻结；
- return、advantage、discount weight 均 stop-gradient；
- 第一版 actor 不输出 duration。

## Critic loss

保留 symlog two-hot value distribution、slow critic 和 replay value learning：

$$
\mathcal L_{\mathrm{value}}^{\mathrm{imag}}
=
-\mathbb E
\left[
w_k
\log p_\xi
\left(
\operatorname{sg}(G_k^\lambda)\mid s_k
\right)
\right],
$$

$$
\mathcal L_{\mathrm{slow}}
=
-\mathbb E
\log p_\xi
\left(
\operatorname{sg}(V_{\mathrm{slow}}(s_k))\mid s_k
\right),
$$

$$
\mathcal L_{\mathrm{value}}^{\mathrm{replay}}
=
-\mathbb E_{\mathcal D}
\log p_\xi
\left(
\operatorname{sg}(G_{k,\mathcal D}^{\lambda})
\mid s_k^{+}
\right),
$$

$$
\mathcal L_{\mathrm{critic}}
=
\mathcal L_{\mathrm{value}}^{\mathrm{imag}}
+
\beta_{\mathrm{slow}}\mathcal L_{\mathrm{slow}}
+
\beta_{\mathrm{replay}}
\mathcal L_{\mathrm{value}}^{\mathrm{replay}}.
$$

是否允许 replay value loss 更新 encoder/RSSM 必须由配置明确控制，并在实验中保持一致。

## 方法优势假设

待验证的优势不是“ODE 在所有任务上都优于 GRU”，而是：

1. 一个共享向量场能处理不同 `delta_t`；
2. 未见 action duration 下具有更好的 interpolation/extrapolation；
3. continuous deterministic flow 与 categorical uncertainty 互补；
4. 以物理时间定义的 reward、continue 和 discount 更符合 SMDP；
5. R2-Dreamer 不训练 observation decoder，可部分抵消 ODE solver 的额外计算；
6. 连续 latent trajectory 可在观测之间查询并分析；
7. Continuous Deter 在单位时长 Euler 下与原 block-GRU update 完全一致，减少替换 Sequence Model 时的结构偏移。

必须同时报告局限：

- 固定步长任务中 ODE 可能没有收益；
- solver 增加 wall-clock、显存和数值误差；
- 接触、碰撞和模式跳变不满足光滑向量场假设；
- latent ODE 不自动具有物理可解释性；
- 只比较 return 不能证明连续时间优势。

## 核心实验矩阵

### Baselines

| 模型         | Sequence model  | `delta_t` 使用方式 | R2 projector |
| ------------ | --------------- | -------------------- | ------------ |
| R2-Dreamer   | 原 block-GRU    | 不使用               | 开启         |
| DT-GRU       | 参数匹配 GRU    | 显式 time embedding  | 开启         |
| ODE-RSSM     | Continuous Deter | 积分区间             | 关闭         |
| CT-R2Dreamer | Continuous Deter | 积分区间             | 开启         |

`DT-GRU` 是必需基线。若 ODE 只优于无时间输入的 GRU，却不能优于 `DT-GRU`，不能把提升归因于 continuous flow。

所有基线应保持：

- 相同 encoder；
- 相同 stochastic state；
- 相近参数量；
- 相同 replay 数据；
- 相同环境交互预算和物理时间范围；
- 相同 reward/continue head 与 actor-critic；
- 相同随机种子集合。

### Environments

第一阶段 DMC Proprio：

- `walker_walk`；
- `cheetah_run`；
- `hopper_hop`；
- `cartpole_swingup`。

第二阶段 DMC Vision：

- `walker_walk`；
- `cheetah_run`；
- `finger_spin`。

第三阶段用 Meta-World 接触任务作为非光滑动力学压力测试，不作为第一阶段开发环境。

### Time regimes

至少包含：

```text
fixed train/fixed test:
    repeat = 2

irregular train/seen test:
    repeat in {1, 2, 4}

irregular train/unseen test:
    train repeat in {1, 2, 4}
    test repeat in {3, 6, 8}

additional robustness:
    random observation gaps
    frame dropping
    short-duration train / long-duration test
    long-duration train / interpolation test
```

必须明确环境基础 control timestep。不同 action repeat 的 episode 应按相同 simulator time 截止，并正确累计 reward。

### Metrics

控制性能：

- episode return；
- return AUC vs environment interactions；
- return AUC vs simulator time；
- seen/unseen `delta_t` zero-shot return；
- observation-gap robustness。

世界模型：

- dynamics KL、representation KL；
- reward NLL；
- continue NLL/AUROC；
- 相同物理时间下的 multi-step prediction error；
- accumulated reward prediction error；
- future encoder-feature prediction error；
- interpolation/extrapolation error。

连续时间专项：

$$
\varepsilon_{\mathrm{semi}}
=
\left\|
\Phi_{\Delta t_1+\Delta t_2}(h)
-
\Phi_{\Delta t_2}
\left(
\Phi_{\Delta t_1}(h)
\right)
\right\|_2.
$$

计算成本：

- wall-clock；
- updates/second；
- environment FPS；
- ODE NFE；
- peak GPU memory；
- solver failure/NaN count。

至少使用 5 个随机种子，报告 mean/standard error，并优先补充 IQM 与 bootstrap confidence interval。

## 消融顺序

按以下顺序推进，避免一次修改多个变量：

1. 原始 R2-Dreamer fixed-`delta_t` baseline；
2. 数据管线加入 `delta_t`，但模型保持不变；
3. 增加参数匹配的 `DT-GRU`；
4. 只将 `Deter` 替换为 Continuous Deter；
5. 比较 ODE-RSSM 与 CT-R2Dreamer；
6. 比较 plain MLP ODE、bounded ODE 与 Continuous Deter；
7. 比较直接 update-gate 速度与额外 rate 参数化；
8. 比较 Tsit5、fixed-step Heun/Euler、不同初始 step/tolerance；
9. 比较 snapshot encoder 与 temporal encoder；
10. 比较 endpoint reward 与 integrated reward rate；
11. 比较 posterior-only R2 与 prior predictive R2；
12. 最后才研究 actor 同时选择 action duration。

## 成功判据

在开始大规模训练前预先确定：

1. fixed-`delta_t` 下 CT-R2Dreamer 不应显著弱于参数匹配基线；
2. irregular 和 unseen-`delta_t` 下应优于原始 R2-Dreamer；
3. 若不能优于 `DT-GRU`，结论只能是 time-conditioning 有效；
4. 改进应同时出现在 world-model prediction 和 control return；
5. 必须报告 wall-clock/NFE，不能隐藏 ODE 的计算成本；
6. 只在固定步长上提升时，不得宣称 continuous-time generalization；
7. 每项主结论至少有 5-seed 统计结果和对应消融。

## 文件地图与拟议修改点

| 文件                          | 当前职责                                     | CTDreamer 拟议修改                                                                                   |
| ----------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `rssm.py`                   | `Deter`、prior、posterior、rollout、KL     | 新增 `ContinuousDeter`；所有 transition 接收 `delta_t`                                            |
| `dreamer.py`                | 世界模型、R2、imagination、actor-critic loss | 时间感知 rollout、按既定条件设计 reward/continue、discount 和 horizon                                |
| `networks.py`               | encoder/decoder/head/projector               | 复用`BlockLinear`；必要时增加 duration embedding 与预测 head                                       |
| `buffer.py`                 | replay sequence 和 latent cache              | 保存、采样并校验`delta_t`                                                                          |
| `trainer.py`                | 环境收集和训练调度                           | 写入物理时间、累计 reward 和 duration 指标                                                           |
| `envs/dmc.py`               | DMC environment wrapper                      | 支持 variable action repeat 和真实`delta_t`                                                        |
| `envs/wrappers.py`          | 通用 observation/reward wrapper              | 保持 transition 时间契约                                                                             |
| `configs/model/_base_.yaml` | 模型与 loss 参数                             | 增加 transition type、ODE solver、速度损失和 time discount 配置                                    |
| `configs/env/*.yaml`        | 环境参数                                     | 增加 base timestep、repeat distribution 和测试 duration                                              |
| `pyproject.toml`            | Python 依赖                                  | 实现阶段加入固定版本的 ODE solver 依赖                                                               |

新增 ODE 代码时优先使用独立小文件，例如 `ode_rssm.py` 或 `models/continuous_deter.py`，避免继续扩大 `rssm.py`。最终命名由实现计划确定。

## 实施阶段

### Phase 0：Baseline 可复现

- 验证 uv/Python 3.11 环境；
- 在 DMC Proprio 跑通原始 R2-Dreamer smoke run；
- 记录 commit、config、seed、return、update speed；
- 不修改算法。

### Phase 1：时间数据管线

- 环境输出 `delta_t`；
- replay 保存和采样 `delta_t`；
- fixed duration 下验证结果不变；
- 添加 shape、reset、reward accumulation 测试。

### Phase 2：Continuous RSSM

- 实现直接复用原 block-GRU core 的 Continuous Deter；
- 验证单位时长 Euler 与原离散 update 数值完全一致；
- 接入 observe、prior rollout 和 imagination；
- 加入 solver/NFE 和 update gate diagnostics；
- 只在 DMC Proprio 做短实验。

当前已完成 Phase 2 的核心实现：`ContinuousDeter` 使用 torchode 的
`AutoDiffAdjoint`，支持 `Tsit5`、`Heun` 和 `Euler`，并已接入
posterior/prior/imagination 的既有 RSSM 调用链。solver 只积分 deterministic state
`h`；`z/action/duration` 通过 torchode `args` 作为区间常量传入，不再拼成 augmented
tensor state。

默认 `transition_train_mode=linear_velocity`：世界模型的 posterior 训练不调用 ODE
solver，而使用

$$
h_{k+1}=h_k+\Delta t_k f_\theta(h_k,z_k,a_k),
$$

并将现有无偏置 R2/InfoNCE projector 作用于 $[0,f_\theta]$，匹配已 detach 的相邻
encoder embedding 线性斜率。跨 episode reset 的区间和零时长区间不参与该速度损失。
在线 `act`、actor imagination、video prediction 和显式 prior rollout 仍默认使用 ODE
求解；训练/推理模式由显式参数选择，不依赖 `module.training`。

线性 transition 和速度损失已验证可被 `torch.compile(fullgraph=True)` 捕获；由于一次
完整 `_cal_grad` 仍包含基于 ODE 的 actor imagination，模型级 `compile` 暂时保持关闭，
solver 使用自己的局部非 fullgraph compile。真实 `delta_t` 尚未接入环境和 replay，缺省
仍按单位时长运行，因此当前结果只能视为 solver-free posterior training 的 transition
replacement，不能视为完整连续时间实验。

### Phase 3：时间感知控制目标

- 修改 reward/continue transition contract；
- 引入 physical-time discount 和 horizon；
- 验证 fixed duration 与旧公式数值对应；
- 比较 R2-Dreamer、DT-GRU、ODE-RSSM、CT-R2Dreamer。

### Phase 4：完整实验

- irregular/unseen duration；
- DMC Vision；
- 5-seed 主结果；
- solver/representation/temporal encoder 消融；
- Meta-World 非光滑压力测试。

## 代码与验证约束

- Python 要求 `>=3.11,<3.12`；当前核心依赖见 `pyproject.toml`；
- 使用当前 uv 虚拟环境，不在系统 Python 中安装依赖；
- 不提交 `.venv/`、训练日志、checkpoint、视频或大规模 replay 数据；
- 新配置必须可由 Hydra 覆盖，不硬编码 task、GPU、solver 或 duration；
- tensor shape 注释沿用当前 `(B, T, ...)` 风格；
- reset、padding 和短 episode 必须显式处理；
- 新模块不得访问 future observation；
- loss 中的 detach/frozen 边界必须写注释并有梯度测试；
- ODE solver failure、NaN 和过大 NFE 不能静默忽略；
- 修改后运行 formatter、静态检查、相关测试和最小 smoke training；
- 实验输出必须记录 git commit、完整 config、seed、环境 base timestep、duration 分布和 solver 配置。

推荐的最小验证层次：

1. vector field 和 ODE transition 的 shape/gradient 单元测试；
2. `delta_t=0` 时状态不变测试；
3. 单位 duration 的线性 transition 与原 GRU-style update 精确对应测试；
4. update gate、速度场和 duration 的有限值测试；
5. 相同 action 下直接积分与分段积分的一致性测试；
6. episode reset 不泄漏状态测试；
7. posterior rollout 与 prior imagination shape 测试；
8. Reward Predictor 不访问终点 observation/posterior 的梯度与依赖测试；
9. Continue Predictor 使用 prior $s_{k+1}$ 的数据流测试；
10. fixed duration smoke training；
11. 两种 duration 的 overfit 小数据测试；
12. DMC Proprio 短预算对照。

## 暂不实施

除非后续方案明确批准，第一阶段不要加入：

- temporal encoder；
- `odeint_event`；
- actor-controlled duration/timer policy；
- continuous relaxation 后统一积分 categorical state；
- observation decoder；
- integrated reward-rate 主模型；
- continuous termination hazard 主模型；
- prior predictive projector 主损失；
- Neural CDE、SDE 或 jump SDE；
- 大规模 Atari/Memory Maze 实验。

## 待用户确认和修改

- 方法最终名称；
- `delta_t` 使用秒还是 base-step 倍数作为存储单位；
- 第一批 DMC task 列表；
- ODE hidden size、block 数和参数匹配规则；
- 是否在后续消融中增加额外 continuous rate 参数化；
- Tsit5 初始 step size、容差与 fixed-step solver 消融范围；
- imagined physical horizon；
- replay value loss 是否继续回传到 world model；
- 主结果统计使用 5 seeds 还是更多；
- 后续是否加入 duration policy。
