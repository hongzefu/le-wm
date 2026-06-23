#!/bin/bash
# LeWM Push-T 全量评测（greatlakes spgpu，单 GPU）
#
# 用途：在 greatlakes 上跑 50-episode Push-T evaluation，复现论文 LeWM 成功率
#       （论文 Fig.6 / Table 5：LeWM = 96.0 ± 2.83%）。
# 本地只做 smoke test 验证管线，全量评测走本脚本。
# 参照 run_slurm_reduceMem.sh 格式 + greatlakes.md 提交规约编写。
#
# 资源说明：
# - eval.py 单 GPU（model.to("cuda")，无 DDP）→ --gpus-per-node=1。
# - --mem=32G：eval 只 cache action/proprio/state（不 cache pixels），内存占用低。
# - --time：由本地 smoke 测速外推后填准（占位见下，>00:30:00 须先经用户确认）。
# - 所有路径写 /nfs/turbo/coe-chaijy-unreplicated/hongzefu/ 绝对路径
#   （greatlakes 计算节点唯一可见的共享盘）。
# - venv 解释器由 uv 装在 NFS（cpython-3.10.19），本机与 greatlakes 两端通用。

#SBATCH --job-name=lewm-pusht-eval
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00   # 全CEM 50ep 本地测速外推 ~20-50min，留足裕量（用户确认）
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/le-wm/output/logs/%x-%j.log
#SBATCH --mail-user=hongzefu@umich.edu
#SBATCH --mail-type=BEGIN,END

set -e
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/le-wm

# STABLEWM_HOME = 仓库根：load_pretrained 把 policy=pusht/lewm 解析到
# checkpoints/pusht/lewm/（weights.pt + config.json）；评测产物写到 pusht/。
# 数据集单独经 cache_dir 指向 data/。
export STABLEWM_HOME=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/le-wm
export HF_HOME=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/le-wm/.hf_cache
export MUJOCO_GL=egl

PYTHON=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/le-wm/.venv/bin/python

# 环境自检：确认 greatlakes 计算节点 driver 支持 cu126（torch 2.12.1+cu126）
echo "=== GPU / driver ==="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$PYTHON" -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available())"

srun --jobid "$SLURM_JOBID" "$PYTHON" eval.py \
    --config-name=pusht.yaml \
    policy=pusht/lewm \
    cache_dir=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/le-wm/data
