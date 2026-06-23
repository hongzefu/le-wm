#!/bin/bash
# # 生产训练脚本（greatlakes spgpu，4×A40）—— reduceMem 变体（低占用 + v3）
#
# ## 2026-06-17 reduceMem 变体：压资源 + 切 dataset-4env-v3 + 全新 run_name（本次）
#
# 本文件是 run_slurm.sh 的「低占用 + v3」分叉，只动资源额度、数据集和命名，模型/训练超参不变。
#
# | 修改 | 原值 → 现值 | 原因 |
# |---|---|---|
# | `--mem` | `160G` → `64G` | wandb run-7ebha22f 实测 rank-0 RSS 峰值仅 9.62GB（160G 只用 ~2.4%）。3 路对抗审查：同款 4-rank DDP 单流 job(51703370,COMPLETED)cgroup Max Memory Used=112GiB，但其中 ~66GB 是可回收 page cache（同 job Max Disk Read=32TiB，137GB 数据集被反复重读 ~233 遍，cache 在 thrash、是高水位假象）；真实硬工作集 ≈ 4×9.62 + pin_memory 锁页 ≈ 40–50GB。cgroup 命中 --mem 先回收 page cache 再 OOM，只要匿名+锁页 <64G 就不被杀。64G 留 ~14–24GB 裕度，残余风险靠 save_every=4 检查点兜底（用户已知 112GiB 旁证、明确选直接用不验证）|
# | `--cpus-per-task` | `10` → `8` | GPU-bound（GPU ~95% util、rank-0 线程 ~24）；4 rank × num_workers=4 = 16 个 seek-based 轻量 dataloader worker，8 核仍喂得饱 |
# | `data.data_root` | `.../dataset-4env-v2` → `.../dataset-4env-v3` | 切 v3（已 verify_dataset.py 校验：子目录仍叫 dataset-token，双流 SigLIP1152+DINOv3 1280 与两路 RMS 齐全）。注意 v3 是 Unmask 任务专一集（任务构成与 v2 不同），kept chunks 50722 vs v2 89108（-43%）|
# | `run_name` | `localDiT_4env_dual_newStat` → `localDiT_4env_v3_dual` | 全新名，overwrite=True 不会清空旧 run 检查点 |
# | `--job-name` | `motion-jepa-newstat` → `motion-jepa-v3` | 区分日志 %x-%j.log |
# | `--time` | 保持 `48:00:00` | v3 chunks 少 43%，~20h 可跑满 40 epoch；48h 裕度足，且 walltime 不影响 mem/cpu 占用诉求 |
#
# ## 2026-06-17 重提：新 loss 监控 + 全新 run_name + 恢复 48h（继承自 run_slurm.sh）
#
# | 修改 | 原值 → 现值 | 原因 |
# |---|---|---|
# | `run_name` | `localDiT_4env_dual` → `localDiT_4env_dual_newStat` | 应用 train.py 新增的 loss 监控（loss_weighted/loss_frac/grad_attr/grad_conflict + _deprecated 镜像 + config/dino_loss_scale）重跑双流；全新名，**不**覆盖旧 run 的 epoch4–28 共 7 个检查点 |
# | `--time` | `24:00:00` → `48:00:00` | 6-15 维护墙已过；双流 ~0.85h/ep，max_epochs=40 ≈ 34h，48h 留 ~14h 裕度跑满 |
# | `--job-name` | `motion-jepa-dual` → `motion-jepa-newstat` | 区分日志 %x-%j.log |
#
# ## 2026-06-14 维护墙 + 切 v2 数据集
#
# | 修改 | 原值 → 现值 | 原因 |
# |---|---|---|
# | `data.data_root` | `.../dataset-4env` → `.../dataset-4env-v2` | 旧 dataset-4env 已用新版 pipeline 干净重建到 v2（norm_stats 原生含两路 RMS，非手工 patch）并延迟删除；v2 自洽含自己的 data-raw+dino-ckpt |
# | `--time` | `48:00:00` → `24:00:00` | 集群维护预留 `SM2026_Maintenance` 从 2026-06-15 04:00 起（MAINT,ALL_NODES 覆盖全部节点含 spgpu），48h 放不进维护前窗口 → `ReqNodeNotAvail` 卡死（job 51766686 实证）。提交时距维护 ~26h50m，24h 留 ~2h50m 启动容差、立即可调度。双流 24h 约 ~12–15 ep；`save_every=4`，维护结束后改 `resume=true` 续跑 |
#
# ## 2026-06-13 双流（SigLIP1152+DINOv31280→2432）
#
# | 修改 | 原值 → 现值 | 原因 |
# |---|---|---|
# | 命令新增 `dino.enabled=True` | （无）→ 显式开 | 接入 DINOv3 第二路特征流：encoder 输入 concat 成 2432 + 新增 dino_decoder + 三重建 loss。config 默认已 true，此处显式自证、防默认漂移 |
# | `training.batch_size` | `16` → `8` | 双流 ~2× 激活（concat 输入 (B,K,256,2432) fp32 单卡就 2.5GB + 第二个 decoder）；batch16 在 A40 44GB 上 OOM 风险，batch8 安全（已先用 run_slurm_check15min.sh 检查显存） |
# | `run_name` | `localDiT_state_image_4env` → `localDiT_4env_dual` | 全新名，**不**清空已有单流 run 的 40 个检查点（overwrite=True 会清空同名目录）|
# | `--job-name` | `motion-jepa-4env` → `motion-jepa-dual` | 区分日志 %x-%j.log |
# | 备注 | max_epochs 仍 40 | 双流更慢（batch 减半→步数翻倍 + 第二 decoder + dino IO），48h 大概跑到 ~25–30ep；靠 save_every=4 检查点取最优 |
#
# ## 2026-06-12 切换 dataset-4env
#
# | 修改 | 原值 → 现值 | 原因 |
# |---|---|---|
# | `data.data_root` | `.../dataset` → `.../dataset-4env` | 换 4 任务 × 100 ep 新数据集（StopCube/ButtonUnmaskSwap/PatternLock/VideoPlaceOrder，600 entries）；`data.dataset=dataset-token` 同名不变 |
# | `run_name` | `localDiT_state_image` → `localDiT_state_image_4env` | 脚本带 overwrite=True，复用旧名会清空 runs/localDiT_state_image/ 旧检查点 |
# | `--time` | `24:00:00` → `48:00:00` | 新 kept chunks 84395 vs 旧 33104 = 2.55×；旧实测 ~14 min/epoch → 推算 ~36 min/epoch × 40 epochs ≈ 23.8h，24h 必超；48h 留 2× 裕度 |
# | `--job-name` | `motion-jepa-localDiT` → `motion-jepa-4env` | 区分日志文件名 %x-%j.log |
#
# ## 2026-06-12 实跑修改记录（job 51699019，与本文件参数完全一致）
#
# | 修改 | 原值 → 现值 | 原因 |
# |---|---|---|
# | `--time` | `4-00:00:00` → `24:00:00` | 4 天 walltime 被 `ReqNodeNotAvail (Reserved for maintenance)` 卡死无法调度；改 24h 后 backfill 立即开跑。实测推算全程仅需 10–13h（40 epochs，~14 min/epoch） |
# | `training.batch_size` | 默认 32 → `16`（每卡） | A40 44GB 上 fp32 batch 32 必 OOM（job 51688448 实证：第一个解码器前向即占满 43.5GB）。batch 16 约 22–26GB，安全。全局有效 batch = 16×4 = 64 |
#
# ## 依赖的环境事实（已验证）
# - `.venv` 解释器已由 uv 正规安装在 NFS（`/nfs/turbo/.../hongzefu/uv-python/`），计算节点可用；
# - 计算节点可直连 api.wandb.ai（job 51692250 实证），wandb 在线模式可用；
# - 提交规约见仓库根目录 `greatlakes.md`。

#SBATCH --job-name=motion-jepa-v3
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/output/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/hongzefu/.bashrc
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA

unset LEROBOT_HOME
unset TRANSFORMERS_CACHE

export WANDB_API_KEY=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

# Auto-detect GPU count
NUM_GPUS=${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
echo "Launching on $NUM_GPUS GPUs"

PYTHON=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/.venv/bin/python

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with DDP (batch_size is PER-GPU; effective = batch_size * NUM_GPUS)
    srun --jobid $SLURM_JOBID bash -c "$PYTHON -m torch.distributed.run \
        --standalone --nproc_per_node=$NUM_GPUS \
        scripts/train.py run_name=localDiT_4env_v3_dual data.data_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/dataset-4env-v3 data.dataset=dataset-token state.enabled=True dino.enabled=True overwrite=True training.batch_size=8 wandb.project=motionjepa wandb.entity=null"
else
    # Single GPU
    srun --jobid $SLURM_JOBID bash -c "$PYTHON scripts/train.py \
        run_name=localDiT_4env_v3_dual data.data_root=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/dataset-4env-v3 data.dataset=dataset-token state.enabled=True dino.enabled=True overwrite=True training.batch_size=8 wandb.project=motionjepa wandb.entity=null"
fi
