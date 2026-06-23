# greatlakes Slurm 提交规约（MotionJEPA）

当端到端验证需要 GPU、本地 GPU 资源不足时，可以 ssh 到 UMich greatlakes 集群提交
slurm job 跑训练 tentative。**以后所有 greatlakes 提交都必须遵守本文件；违反任何
"硬规则"前必须先与用户确认，不可静默放宽。**

## 登录认证（硬规则，不可静默放宽）

**优先复用 ControlMaster 主连接:认证一次，8h 内所有 ssh 操作（提交/查询）免认证、免手机；
没有 master 才需建立（仅这一次走 Okta 2FA）。** ssh 二次验证已从 Duo 迁移到 **Okta Verify**。
**仅在"建立主连接"那一次需要验证；此时必须先问用户用哪种方式（6 位 TOTP 码 / 留空触发
push + 数字匹配，强烈推荐 TOTP），不要默认或复用上次选择。** 先 `ssh -O check greatlakes`
判断 master 是否存活：存活就直接干活、不必问验证方式。详见下方「ssh 提交流程（ControlMaster
复用，Okta Verify）」。

## 资源约束（硬规则，不可静默放宽）

- `--account=chaijy2`：不能切换到任何其他 account（即便看到别的 account 可绕过排队也不行）；
- `--partition=spgpu`：不能用其他 partition（如 standard / gpu / largemem）；
- `--nodes=1` + `--ntasks-per-node=1`：永远只提单 node 单 task；
- `--gpus-per-node` ≤ 2 且 `--time` ≤ 00:30:00：日常调试默认 1–2 GPU、20–30 分钟内；
  如确实需要更长时间或更多 GPU，必须显式告知用户并征得确认，不可静默放宽；
- 默认 `--mem=32G`（足以跑 tentative）；**实测 `--qos=interactive` 在 chaijy2/spgpu 下报
  `Invalid qos specification`，不要再用** —— 默认不指定 qos 即可；遇到 `(AssocGrpMemLimit)`
  时先降 `--mem`，正确的 qos 名待用 `sacctmgr show assoc user=hongzefu format=qos`
  （或 `sacctmgr show qos`）查清；`(AssocGrpGRES)` 表示 chaijy2 账户 GPU 配额已被组内
  其他用户占满，此时只能等他们的 job 退出，不可换 account / partition 绕开。

生产长训（如 `scripts/train-hongzefu/run_slurm.sh` 的 4 GPU / 4 天配置）超出上述
调试限制，提交前必须由用户显式确认。

## 路径可见性（硬规则，不可静默放宽）

greatlakes 计算节点唯一能看到的共享路径是 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/`。
本机 `sled-vail` 的其它路径在 slurm 节点上全部不可见，写进脚本会立刻
`No such file or directory`：

- `/home/...`（本机 home，包括 `~/...`）—— 注意 greatlakes 有自己的 `/home/hongzefu`，
  与本机 home 是**两个不同的目录**，内容不互通；
- `/data/...`（本机 data 盘）；
- `/tmp/...`、`/var/tmp/...`（计算节点的 `/tmp` 是节点自己的，跟本机无关）；
- 任何不以 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 开头的本机绝对路径；
- 相对路径（slurm job 的 cwd 不是本机当前目录；脚本内先 `cd` 到 NFS 绝对路径再用相对路径可以）。

所有 slurm 脚本里出现的路径——`data.data_root`、checkpoint / `runs_root` 目录、
`--output` 日志、yaml / config、norm_stats、wandb dir、`cd` 的工作目录、python
解释器路径——都必须落在 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 下，并写成绝对路径。

如果要用的数据 / 产物当前还在本机非共享路径，提交 slurm 前必须先 rsync 到
`/nfs/turbo/coe-chaijy-unreplicated/hongzefu/` 下；不要尝试 mount / symlink / 把本机
路径硬塞给 sbatch。

## venv 可移植性（硬规则：解释器必须由 uv 安装在 NFS 上）

uv 默认把 managed Python 装在本机 home（`/home/hongzefu/.local/share/uv/python/`），
`.venv/bin/python` symlink 过去，在 greatlakes 计算节点上是**死链**。**禁止用手动
重链 / 改 pyvenv.cfg 的方式修补**；正规做法是把 uv 的解释器安装目录放到 NFS：

```bash
# 1. 解释器装到 NFS（已装好 3.11.14，重装其它版本时同样带这个环境变量）
UV_PYTHON_INSTALL_DIR=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python \
    uv python install 3.11.14

# 2. 重建 venv 时显式指定 NFS 解释器的绝对路径（防止 uv 抓回本机 home 的解释器）
uv venv --python /nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
UV_LINK_MODE=copy uv sync    # cache 在本机盘、venv 在 NFS，跨设备必须 copy
```

这样 `pyvenv.cfg` 的 `home` 与 `bin/python` symlink 天然落在 NFS，本机与 greatlakes
双端可用，无任何手术。验证命令：
`.venv/bin/python -c "import torch; print(torch.__version__)"`。

## ssh 提交流程（ControlMaster 复用，Okta Verify）

greatlakes 登录节点（`greatlakes.arc-ts.umich.edu`）拒绝纯密码 ssh，需要
`keyboard-interactive`（密码 + **Okta Verify** 2FA，已从旧 Duo 迁移）。**核心策略:没有
master 就建立 master——认证一次，8h 内所有提交/查询免认证、免手机。**

`~/.ssh/config` 已为 `greatlakes` 配好 `ControlMaster auto` + `ControlPersist 8h`
（`ControlPath ~/.ssh/cm-%r@%h:%p`）。OpenSSH 复用原理:主连接认证一次后建立 control
socket，之后 `ssh greatlakes <cmd>` 直接复用已认证通道、**不再发起任何 SSH 认证握手**，
因此零密码零 MFA 零手机（slave 不认证，与服务器 MFA 策略无关）。

现成提交器:**`scripts/train-hongzefu/gl_submit.py`**（自包含，纯系统 ssh + ControlMaster，
不再用 paramiko）——逻辑就是"没 master 就建 master，再经系统 ssh 复用提交":

- **master 存活** → 直接提交，**无需任何凭据、不必问验证方式**；
- **master 不存活** → 用 `GLPW`(+`GLOTP`) 经 `pexpect` 驱动系统 ssh 建立主连接，再提交。

无参数默认提交 `run_slurm_test20s.sh` 并打印 squeue；也可传一条远程命令当参数
（如 `"squeue -u hongzefu"`）——会自动前置 `cd {REPO} &&`（远程 cwd 是 home，相对路径
否则找不到脚本）。

**密码安全（硬规则）:绝不把密码写入任何文件、commit、CLAUDE.md 或对话历史。凭据由
用户即时提供，仅经临时环境变量 `GLPW`（建 master 时必需）/ `GLOTP`（可选 6 位 TOTP，推荐）
传入，用完立即 unset、绝不持久化。**

### 标准流程

1. **先查 master**（纯本地 socket 检查，不出网、不需凭据、不需 sandbox）:
   ```bash
   ssh -O check greatlakes   # exit 0 = 存活可复用;非 0(255) = 需先建
   ```
2. **存活 → 直接提交/查询**（零认证零手机，不必问验证方式）:
   ```bash
   uv run --no-project --with pexpect python scripts/train-hongzefu/gl_submit.py "squeue -u hongzefu"
   uv run --no-project --with pexpect python scripts/train-hongzefu/gl_submit.py   # 默认提交 test20s
   ```
3. **不存活 → 建主连接**（**仅此步需要验证方式**，推荐 TOTP、给一次码无需手机匹配）。
   gl_submit 在无 master 时会自动用凭据建连后再提交;也可设好凭据直接跑:
   ```bash
   export GLPW='<密码>' GLOTP='<当前6位码>'
   uv run --no-project --with pexpect python scripts/train-hongzefu/gl_submit.py "<命令>"
   unset GLPW GLOTP
   ```
   建好后当天所有提交/查询走第 2 步、免认证;8h 过期后重建。（skill `greatlakes-usage`
   的 `gl_connect.py` 也能单独建 master，与此共用同一个 socket。）

### Okta 两条路 —— 仅"建主连接"那一次需要，且每次先问用户用哪种，不默认 / 复用上次

建 master 时 keyboard-interactive 的 prompt 依次是 `Password:` 和
`Okta passcode (leave blank to initiate a push):`:

1. **6 位 TOTP 码（强烈推荐，最可靠、无推送时序风险）**:从 Okta Verify app 读当前 6 位码
   填 `GLOTP`。码每 30s 刷新且一次性，**拿到立刻发起**。
2. **留空触发 push + number challenge（不推荐，除非按「push 修法」处理）**:不设 GLOTP，
   SSH 端依次显示 `Successfully initiated Okta push` → `The correct answer is N`（用户在手机
   Okta Verify 选中 N）→ `Press enter to continue:`（**approve 后还要再按一次回车，模块才校验**）。
   **2026-06-19 摸清真根因:以前判定的"转达数字超时、错过 ~60s 窗口"不准确——真正原因是
   `gl_submit.py` / `gl_connect.py` / `gl_master.py` 的 pexpect 模式表里没有 `Press enter to
   continue` 这一条,发完空 passcode 就一直 `expect()` 干等到 TIMEOUT，根本没按那下回车，于是
   "看起来卡死/超时"。** 用这些现成驱动走 push 必挂;要么用 TOTP，要么按下方「push 修法」用增强
   驱动。**能用 TOTP 就用 TOTP（无回车握手、无数字转达）。**

pexpect 建连要点（见 gl_submit.py / skill 的 `gl_master.py`）:匹配 `Password:` 填 GLPW、
`Okta passcode` 填 GLOTP（空则触发 push）、首次 host key 提示自动答 `yes`，认证通过的标志是
远端回显 `CONNECTED_OK_MARKER`。本机无 expect/sshpass，故用 `uv run --with pexpect` 临时
拉 pexpect 驱动系统 ssh，无需 sudo 装包。**注意:这些现成驱动只覆盖 TOTP 路径——模式表里
没有 push 的 `Press enter to continue`，故走 push 会卡死（见上）。push 必须用下方增强驱动。**

#### push 修法（2026-06-19 实测一次过；仅当用户坚持用 push 时才需要）

若必须走 push，用一个增强版 pexpect 驱动（一次性脚本即可，密码仍只经 `GLPW` 不落盘），相对
现成驱动多做三件事:

1. **补两条 pexpect 模式**:`correct answer is (\d+)`（捕获数字 N，实时打印/落盘）+
   `Press enter to continue`（匹配到就 `sendline("")` 按回车，模块这才去校验 approval）。缺第二条
   就是"卡死"的全部原因。
2. **服务器输出实时落盘 + 把日志路径直接给用户**:`child.logfile_read = <每写即 flush 的文件>`
   （只记服务器→本地，不含密码），并把该日志绝对路径丢给用户自己 `tail -f` 看 N——别靠转述，
   转述既慢又易漏（数字行常是单字符、易被过滤器吃掉）。
3. **回车用 sentinel 文件握手**:驱动在 `Press enter to continue` 处阻塞轮询一个 sentinel
   文件（如 `/tmp/gl_push_go`），用户在手机点完 N、回话确认后再 `touch` 该文件放行、然后才按
   回车——**避免 approve 之前就按回车导致单次校验失败**。

完成后 `ssh -O check greatlakes` 应为 `Master running`，主连接建好，后续提交/查询零认证。
**结论不变:push 全程要"看数字→点→确认→放行回车"四步握手，TOTP 一步到位，优先 TOTP。**

### 已知坑

- **远程命令 cwd 是 greatlakes 的 `/home/hongzefu`，不是 REPO**:gl_submit 已对自定义命令
  自动前置 `cd {REPO} &&`，直接传 `"sbatch scripts/train-hongzefu/run_slurm.sh && squeue ..."`
  即可，**不要再自己写 cd**（会变成双 cd，虽无害但多余）。squeue 的 `-o '...'` 单引号在外层
  参数里是字面、远程 bash 才解析。
- **spgpu 强制至少 1 GPU**:提交 `--gpus-per-node=0` 报 `QOSMinGRES` / `Batch job
  submission failed`（2026-06-17 实证）。纯 CPU 的 sleep 测试 job 也必须带 `--gpus-per-node=1`。

## 调试 slurm 脚本

调试脚本放在 `scripts/train-hongzefu/`，与生产 `run_slurm.sh` 分开命名
（如 `run_slurm_debug5min.sh`）。调试脚本要点：

- 独立 `run_name`（绝不复用生产 run_name —— `overwrite=True` 会清空 `runs/<run_name>/`）；
- `training.compile=false`（torch.compile 首编几分钟，短 job 全耗在编译上）；
- `wandb.enabled=false`、`training.max_epochs=1`；
- 不 `source` 任何 home 下的 rc 文件，python 用 NFS 绝对路径。

常用样板：

```
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
# 注：不要加 --qos=interactive（chaijy2/spgpu 报 Invalid qos specification），用默认 qos
#SBATCH --output=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA/output/logs/%x-%j.log
```

## PENDING 状态读法

- `(Priority)` → 正常排队等调度，等就行；
- `(AssocGrpMemLimit)` → chaijy2 总 mem 配额满了，先降 `--mem`（`--qos=interactive` 实测无效，别用）；
- `(AssocGrpGRES)` → chaijy2 总 GPU 配额满了，只能等组内其他用户的 job 退出，不可换 account；
- `(Resources)` → spgpu 全集群节点都被占，等就行。
