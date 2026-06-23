# LeWorldModel (LeWM)
### 从像素端到端稳定训练的联合嵌入预测架构（JEPA）

> 本文档是 [README.md](README.md) 的中文补充版，重点讲清楚 **模型架构、输入/输出、数据流**，并配有 ASCII 结构图。安装、数据下载、检查点加载等运维细节请以英文 [README.md](README.md) 为准。

作者：[Lucas Maes*](https://x.com/lucasmaes_)、[Quentin Le Lidec*](https://quentinll.github.io/)、[Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr)、[Yann LeCun](https://yann.lecun.com/)、[Randall Balestriero](https://randallbalestriero.github.io/)

**[ [论文](https://arxiv.org/pdf/2603.19312v1) | [检查点与数据](https://huggingface.co/collections/quentinll/lewm) | [项目主页](https://le-wm.github.io/) ]**

---

## 1. 这是什么

LeWM 是一个 **世界模型（world model）**：它不在像素空间里预测「下一帧长什么样」，而是在一个紧凑的 **隐空间（latent space）** 里预测「下一时刻的表征（embedding）」。这类架构称为 **JEPA（Joint Embedding Predictive Architecture，联合嵌入预测架构）**。

LeWM 的核心卖点是 **稳定且极简**：

- **完全从原始像素端到端训练**，不需要预训练编码器、不需要 EMA（指数滑动平均）目标网络、不需要额外监督信号。
- **只有两项损失**：一项「下一步嵌入预测损失」，一项「让隐空间服从各向同性高斯分布的正则项」。相比唯一已有的端到端方案，可调的损失超参数从 6 个降到 1 个。
- **轻量**：约 **15M** 可训练参数，单张 GPU 几小时即可训练完。
- **规划快**：用于 MPC 规划时，比基于基础模型（foundation model）的世界模型快最多 **48×**，同时在 2D / 3D 控制任务上保持竞争力。

训练好后，模型本身不是策略（policy），而是一个 **代价函数（cost model）**：给定当前观测、目标图像和一串候选动作，它在隐空间里推演（rollout）未来，输出每条动作序列的代价。外部的规划器（CEM 或梯度法）据此优化动作，实现 **模型预测控制（MPC）**。

---

## 2. 模型架构总览

LeWM 由 5 个子模块组成（[jepa.py](jepa.py) 里的 `JEPA` 类），其中编码器是一个 ViT，预测器是一个带动作条件化的因果 Transformer：

```
                         LeWorldModel (LeWM) —— 训练前向
                         ════════════════════════════════

   观测 pixels (B, T, 3, 224, 224)                动作 action (B, T, A)
              │                                            │
              │  rearrange:  b t ... -> (b t) ...          │
              ▼                                            ▼
   ┌────────────────────────┐                  ┌──────────────────────────┐
   │  encoder  (ViT-tiny)    │                  │  action_encoder           │
   │  patch=14, 224×224      │                  │  (Embedder)               │
   │  取 CLS token            │                  │  Conv1d(k=1) + MLP        │
   └───────────┬────────────┘                  └─────────────┬────────────┘
               │ (B*T, 192)                                   │
               ▼                                              │
   ┌────────────────────────┐                                │
   │  projector  (MLP+BN)    │                                │
   └───────────┬────────────┘                                │
               │  emb  (B, T, 192)                            │ act_emb (B, T, 192)
               │                                              │
               └───────────────────┬──────────────────────────┘
                                   │  x = emb,  条件 c = act_emb
                                   ▼
                    ┌──────────────────────────────────┐
                    │  predictor  (ARPredictor)         │
                    │  6 层因果 Transformer              │
                    │  AdaLN-zero 用 action 做条件化      │
                    │  + 可学习位置编码                   │
                    └──────────────────┬───────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  pred_proj  (MLP+BN)    │
                          └───────────┬────────────┘
                                      │ pred_emb (B, T, 192)
                                      ▼
        ┌──────────────────────────────────────────────────────┐
        │  Loss = pred_loss  +  λ · sigreg_loss   (λ = 0.09)     │
        │    pred_loss   = MSE( pred_emb ,  tgt_emb )            │
        │    sigreg_loss = SIGReg( emb.transpose(0,1) )          │
        │                  → 各向同性高斯正则，防表征坍塌          │
        └──────────────────────────────────────────────────────┘
```

各子模块一览（默认配置见 [config/train/model/lewm.yaml](config/train/model/lewm.yaml)）：

| 子模块 | 实现 | 作用 | 关键超参 |
|---|---|---|---|
| `encoder` | `vit_hf`（HuggingFace ViT） | 把每帧图像编码成一个全局向量（取 **CLS token**，不用 patch tokens） | size=`tiny`(隐维 192)，patch=14，224×224，**非预训练** |
| `projector` | `MLP`（带 BatchNorm1d） | 对 CLS 嵌入再投影一次，得到 `emb` | 192→2048→192 |
| `action_encoder` | `Embedder`（Conv1d k=1 + MLP） | 把（叠帧后的）动作编码成 `act_emb` | input_dim = `frameskip × 环境动作维`，输出 192 |
| `predictor` | `ARPredictor`（因果 Transformer） | 给定历史 `emb` 和动作条件，自回归预测下一步 `emb` | depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1 |
| `pred_proj` | `MLP`（带 BatchNorm1d） | 对预测结果再投影一次 | 192→2048→192 |

> 说明：`D = embed_dim = 192` 来自 ViT-tiny 的隐藏维度；`A = frameskip × env_action_dim`（PushT 中环境动作是 2 维、frameskip=5，故 A=10）。

### 预测器为什么是「条件化 Transformer」

`predictor` 用的是 **AdaLN-zero** 条件化的 Transformer 块（[module.py](module.py) 中的 `ConditionalBlock`）：动作嵌入 `act_emb` 不是拼接进序列，而是作为条件 `c`，经一个 `SiLU + Linear` 生成 6 组调制参数（shift / scale / gate，分别作用于注意力和 MLP 两个子层）。`adaLN_modulation` 末层权重和偏置初始化为 0（即 "zero" 初始化），使训练初期每个块近似恒等映射，从而更稳定。注意力使用 **因果掩码**（`is_causal=True`），保证位置 t 只能看到 ≤ t 的历史，符合自回归预测的设定。

---

## 3. 输入与输出

### 3.1 训练阶段

训练前向逻辑在 [train.py](train.py) 的 `lejepa_forward`。一个 batch 里每条样本是一小段连续轨迹，长度 `T = history_size + num_preds = 3 + 1 = 4`。

**输入（一个 `batch` 字典）：**

| 键 | 形状 | 含义 |
|---|---|---|
| `pixels` | `(B, T, 3, 224, 224)` | 连续 T 帧的 RGB 观测（已按 ImageNet 统计归一化、resize 到 224） |
| `action` | `(B, T, A)` | 对应的动作（已做 frameskip 叠帧 + z-score 归一化；序列边界的 NaN 会被置 0） |
| `proprio` / `state` 等 | `(B, T, ...)` | 可选的本体感受/状态列，按配置加载，训练损失里不直接使用 |

**中间产物：**

- `emb = encode(pixels)` → `(B, T, 192)`
- `act_emb = action_encoder(action)` → `(B, T, 192)`

**输出（损失）：**

| 键 | 计算 | 含义 |
|---|---|---|
| `pred_loss` | `MSE(predict(emb[:, :3], act_emb[:, :3]), emb[:, 1:])` | 下一步嵌入预测损失 |
| `sigreg_loss` | `SIGReg(emb.transpose(0,1))` | 各向同性高斯正则（防坍塌）；输入先由 `(B,T,D)` 转置成 `(T,B,D)` |
| `loss` | `pred_loss + 0.09 × sigreg_loss` | 总损失，用于反向传播 |

### 3.2 推理 / 规划阶段

推理入口是 [jepa.py](jepa.py) 的 `get_cost`，它被规划器（CEM / 梯度法）反复调用。

**输入：**

| 名称 | 形状 | 含义 |
|---|---|---|
| `info_dict["pixels"]` | `(B, S, T, 3, 224, 224)` | 初始观测（含历史帧），S 是动作候选采样数 |
| `info_dict["goal"]` | `(B, S, T, 3, 224, 224)` | 目标图像序列（与 `pixels` 同样带 T 维；编码前按 `[:, 0]` 取样） |
| `action_candidates` | `(B, S, T, A)` | 待评估的候选动作序列 |

**输出：**

| 名称 | 形状 | 含义 |
|---|---|---|
| `cost` | `(B, S)` | 每条候选动作序列的代价（rollout 末步嵌入与目标嵌入的逐元素平方误差之和 SSE） |

规划器拿到 `cost` 后优化动作；外层是 receding-horizon（滚动时域）的 MPC 闭环。

---

## 4. 训练时的数据流（自回归错位）

训练并不是「拿 3 帧预测第 4 帧」这么简单，而是 **每个时间位置都预测它的下一步**，借因果注意力一次性算完——这正是 `tgt_emb = emb[:, n_preds:]` 这一行「错位一格」的含义（`n_preds=1`）。

```
   时间步:        t=0      t=1      t=2      t=3
   ─────────────────────────────────────────────────
   emb (编码):   e0       e1       e2       e3
   act_emb:      a0       a1       a2        ·

   上下文输入     [ e0       e1       e2 ]            ← emb[:, :3]
   (history=3):  [ a0       a1       a2 ]            ← act_emb[:, :3]
                    │        │        │
              因果 Transformer (位置 t 仅看 ≤ t)
                    │        │        │
   预测 pred_emb:   ê1       ê2       ê3              ← predict(...)
                    ║        ║        ║   (逐位置 MSE)
   预测目标 tgt:    e1       e2       e3              ← emb[:, 1:]
```

要点：

- **目标是「自己编码出来的下一帧嵌入」**，不是像素、不是另一个目标网络的输出——这就是 JEPA 「在隐空间里自我预测」的本质。
- 编码器、预测器、两个投影头 **全部一起端到端更新**（不冻结、无 stop-gradient 到目标网络）。防坍塌完全靠 `SIGReg` 正则项，而不是靠 EMA 或负样本。

---

## 5. 推理时的自回归推演与 MPC 闭环

`rollout`（[jepa.py](jepa.py)）从初始几帧出发，用预测器 **自回归地** 一步步推演未来嵌入；每步只截取最近 `history_size=3` 步作为输入窗口。`get_cost` 用推演末步嵌入和目标嵌入的距离作为代价，交给规划器优化动作。

```
   规划器(CEM / Adam)
   提出候选动作 (B,S,T,A)
            │
            ▼
   ┌──────────────────────────────────────────────────────────┐
   │  rollout:  编码初始帧 → e0..e_{H-1}                          │
   │                                                            │
   │   循环 n 步：                                                │
   │     窗口 = emb[-3:],  动作窗口 = act_emb[-3:]                 │
   │     ê_next = predict(窗口, 动作窗口)[:, -1:]                  │
   │     emb ← concat(emb, ê_next)        ← 自回归喂回            │
   │     act ← concat(act, 下一个候选动作)                         │
   │   循环后再 predict 一次收尾 → 末步嵌入 ê_last                 │
   │   (rollout 共产生 n+1 个预测嵌入，jepa.py:99-104)             │
   └───────────────────────────┬──────────────────────────────┘
                               │  predicted_emb (B,S,...,192)
                               ▼
   ┌──────────────────────────────────────────────────────────┐
   │  criterion:  cost = SSE( 预测末步 ,  目标嵌入末步 )           │  → (B, S)
   └───────────────────────────┬──────────────────────────────┘
                               ▼
                     规划器据 cost 更新候选动作
                     （CEM 取 topk 重采样 / Adam 反传梯度）
                               │
                               ▼
                 取最优动作的前若干步执行 → 环境前进 → 滚动时域重规划
```

可选的两种规划器（[config/eval/solver/](config/eval/solver/)）：

- **CEM**（`cem.yaml`）：交叉熵法，采样 300 条、迭代 30 步、保留 topk=30。无梯度、对非光滑代价稳健。
- **梯度法**（`adam.yaml`）：对动作直接用 AdamW（lr=0.1）反传 30 步。利用世界模型可微的优势。

---

## 6. 评估任务与 Simulator

LeWM 在 **4 个连续控制任务**上做规划评估,覆盖 **2D / 3D**、**导航 / 操控 / 触达**,动作空间全部连续。这些环境都经 [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 封装,以 **Gymnasium** 标准接口暴露——代码里 `env_name` 一律是 `swm/...`(见 [config/eval/](config/eval/))。其中 **OGBench-Cube** 与 **Reacher** 的底层物理由 **MuJoCo** 驱动([eval.py](eval.py) 顶部 `os.environ["MUJOCO_GL"] = "egl"` 即为其离屏渲染);Push-T / Two-Room 则是 2D 环境。

| 任务 | 维度 / 类型 | 任务定义 | 底层 simulator / 来源 | 代码 `env_name`(配置) |
|---|---|---|---|---|
| **Two-Room** | 2D · 导航 | 两个房间被一堵带单门的墙隔开;agent(红点)从一室随机起点出发,**穿过门**到达另一室的随机目标位置 | Sobal et al.(PLDM)提出的轻量自定义 2D 导航环境 | `swm/TwoRoom-v1`([tworoom.yaml](config/eval/tworoom.yaml)) |
| **Push-T** | 2D · 操控 | agent(蓝点)**只能推**,把一个 **T 形 block** 推到与目标配置对齐 | pymunk 2D 刚体物理(沿用 DINO-WM 的 PushT 环境) | `swm/PushT-v1`([pusht.yaml](config/eval/pusht.yaml)) |
| **OGBench-Cube** | **3D** · 机械臂操控 | 带末端执行器的机械臂**抓起 cube 放到目标位置**;仅用 single-cube 变体 | **MuJoCo**;OGBench(Park et al.) | `swm/OGBCube-v0`([cube.yaml](config/eval/cube.yaml)) |
| **Reacher** | 2D 平面 · 连续控制 | 控制**双关节机械臂**,使关节与目标配置完美对齐,以触达 2D 平面内的目标(`task: qpos_match`) | **MuJoCo**;DeepMind Control Suite | `swm/ReacherDMControl-v0`([reacher.yaml](config/eval/reacher.yaml)) |

**各任务的数据集与评估预算**(数据集来自 [HuggingFace 合集](https://huggingface.co/collections/quentinll/lewm),均训练 10 epochs):

| 任务 | 训练数据集 | 数据规模 / 采集策略 | eval budget | goal 采样间隔 |
|---|---|---|---|---|
| Two-Room | `tworoom` | 10,000 条 × 均 92 步;噪声启发式(先奔门、再奔目标) | 150 步 | 100 步后 |
| Push-T | `pusht_expert_train` | 20,000 条专家 × 均 196 步(同 DINO-WM) | 50 步 | 25 步后 |
| OGBench-Cube | `ogbench/cube_single_expert` | 10,000 条 × 200 步;benchmark 自带启发式 | 50 步 | 25 步后 |
| Reacher | `dmc/reacher_random` | 10,000 条 × 200 步;Soft Actor-Critic 策略 | 50 步 | 25 步后 |

**评估方式**:统一用第 5 章的 **goal-conditioned MPC** 闭环——从离线数据集随机采一条轨迹的某状态作为**初始状态**,把同一轨迹 goal 间隔步之后的状态作为**目标**(保证目标可达、与数据动力学一致),再用世界模型规划动作去逼近目标。

> **几点说明**
> - **规划(planning)实验用全部 4 个任务**(论文 Fig. 6);而**物理量探针(probing)**与**违背预期 / 意外性(surprise)实验只用 Two-Room、Push-T、OGBench-Cube 三个**。
> - 上表 budget / goal 间隔为**论文 F.1 的设置**;仓库 [config/eval/](config/eval/) 各 yaml 的默认值统一是 `eval_budget=50`、`goal_offset_steps=25`(Two-Room 论文实际用 150 / 100,复现时需相应调大)。
> - **单任务训练,而非 multi-task**:每个环境**各训一个独立 LeWM**([train.py](train.py) 只加载单一数据集,`python train.py data=<env>`),HuggingFace 上每环境对应一个 ckpt repo(`lewm-pusht/-cube/-tworooms/-reacher`,4 个全部可用)。论文所说的"统一"指**同一套架构 + 同一组超参**适用所有环境("we keep the hyperparameters fixed across all environments"),**而非共享一组权重**——各环境动作维不同,`action_encoder` 输入维即按该环境动作空间设定。
> - **统一实现**:训练用 [stable-pretraining](https://github.com/galilai-group/stable-pretraining),评估用 PyTorch + Gymnasium;论文全部实验跑在**单张 NVIDIA L40S GPU** 上。

---

## 7. 两项损失详解

LeWM 全部稳定性来自这两项（[train.py](train.py) `lejepa_forward`）：

**(1) 下一步嵌入预测损失 `pred_loss`**

```
pred_loss = mean( ( predict(ctx_emb, ctx_act) − tgt_emb )² )
```

让预测器学会「给定历史观测嵌入 + 动作，推断下一时刻的观测嵌入」——这就是世界模型的核心能力。

**(2) Sketch Isotropic Gaussian Regularizer `SIGReg`**（[module.py](module.py) `SIGReg`）

这是防止 **表征坍塌**（所有输入被编码成同一个点）的关键，也是 LeWM 能去掉 EMA / 负样本 / stop-gradient 的原因。训练时先把 `emb` 从 `(B, T, D)` 转置成 `(T, B, D)` 再送入（[train.py](train.py) 中 `self.sigreg(emb.transpose(0, 1))`），其做法：

1. 随机抽 `num_proj=1024` 个单位方向，把嵌入投影到一维；
2. **在每个时间步上、跨 batch 内样本** 计算 **Epps–Pulley 统计量**（用一维投影的经验特征函数 `cos/sin` 均值，与标准高斯特征函数 `exp(−t²/2)` 比对，在 `knots=17` 个节点上做数值积分，并按样本数缩放）；
3. 最后对所有随机投影与时间步取平均；统计量越小，说明嵌入分布越接近 **各向同性标准高斯**。

把嵌入「推向高斯」既保证了各维度被充分利用（不坍塌），又给隐空间一个良态的几何结构。整份实现单卡即可运行，无需跨卡通信。

总损失：`loss = pred_loss + λ · sigreg_loss`，其中 `λ = 0.09`（[config/train/lewm.yaml](config/train/lewm.yaml)）——**唯一需要调的损失超参数**。

---

## 8. 代码结构

| 文件 | 内容 |
|---|---|
| [jepa.py](jepa.py) | `JEPA` 主类：`encode` / `predict`（训练）+ `rollout` / `criterion` / `get_cost`（推理规划） |
| [module.py](module.py) | 网络组件：`SIGReg`、`Attention`、`Block` / `ConditionalBlock`（AdaLN-zero）、`Transformer`、`ARPredictor`、`Embedder`、`MLP` |
| [train.py](train.py) | 训练入口：Hydra 配置、数据加载/归一化、`lejepa_forward` 前向与损失、Lightning 训练循环 |
| [eval.py](eval.py) | 评估入口：构建环境、加载检查点、用 `WorldModelPolicy` 做 MPC 规划并统计指标 |
| [utils.py](utils.py) | 图像预处理、列 z-score 归一化、按 epoch 存检查点的回调 |
| [config/train/](config/train/) | 训练配置（模型结构、数据、优化器、损失权重等） |
| [config/eval/](config/eval/) | 评估配置（环境、规划器、规划时域等） |

本仓库刻意做得很薄：环境管理、规划、评估复用 [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)，训练框架复用 [stable-pretraining](https://github.com/galilai-group/stable-pretraining)，仓库本身只保留核心贡献——**模型架构 + 训练目标**。

---

## 9. 关键超参数（默认）

来自 [config/train/lewm.yaml](config/train/lewm.yaml) 与 [config/train/model/lewm.yaml](config/train/model/lewm.yaml)：

| 名称 | 值 | 含义 |
|---|---|---|
| `img_size` | 224 | 输入图像边长 |
| `embed_dim` | 192 | 隐空间维度（= ViT-tiny 隐藏维） |
| `history_size` | 3 | 上下文/推演窗口长度 |
| `num_preds` | 1 | 预测错位步数（自回归预测下一步） |
| predictor `depth/heads` | 6 / 16 | 预测器层数与注意力头数 |
| `frameskip` | 5 | 动作叠帧步长 |
| 优化器 | AdamW, lr=5e-5, wd=1e-3 | + 线性 warmup 余弦退火 |
| `batch_size` | 128 | 批大小 |
| `max_epochs` | 100 | 训练轮数 |
| 精度 | bf16 | 混合精度 |
| `sigreg.weight` (λ) | 0.09 | 高斯正则权重（唯一损失超参） |

---

## 10. 快速上手（详见 [README.md](README.md)）

```bash
# 安装
uv venv --python=3.10 && source .venv/bin/activate
uv pip install stable-worldmodel[train,env]

# 训练（数据放在 $STABLEWM_HOME，默认 ~/.stable-wm/）
python train.py data=pusht

# 评估（policy 填检查点相对路径，去掉 _object.ckpt 后缀）
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

支持的任务：`tworoom`、`pusht`（2D）、`reacher`（2D 平面，DMC）与 `cube`（OGBench，**3D**）——定义与 simulator 详见前文「6. 评估任务与 Simulator」。各环境的预训练检查点见 [HuggingFace 合集](https://huggingface.co/collections/quentinll/lewm)。

---

## 11. 引用

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

问题与合作：欢迎提 [issue](https://github.com/lucas-maes/le-wm/issues)，或联系 `lucas.maes@mila.quebec`。
