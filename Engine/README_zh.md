# Avril V3.0

V3.0 在完整搜索与精确重拟合之后增加“有意义候选漏斗”：目录已知
信号、经真实低点重合证明的谐波/事件别名、事件方向不一致，以及由
数据缺口边缘单次异常主导的信号都会保留在审计记录中，但不再浪费
红噪声、物理和像素排假计算。只有未解释、重复出现且不由缺口主导
的候选进入完整排假。未经完整排假的结果仍只能称为候选，不能称为
已确认行星。

Avril 是一套保留完整证据链的 TESS 凌日候选搜索引擎。CPU/Mac 和 NVIDIA CUDA
版本使用相同的科学参数、候选晋级规则和最终排假，只切换数值执行后端。

最方便的安装方式：把算法 ZIP 和网页 ZIP 解压在同一个目录。Mac 第一次双击
`Install Avril.command`，以后双击 `Start Avril.command`；Windows 第一次双击
`Install-Avril.bat`，以后双击 `Start-Avril.bat`。网页已经预构建，不需要朋友安装
Node.js；首次安装只会建立本地 Python 环境，计算与 FITS 数据都留在本机。

```bash
./setup_avril.sh cpu
source .venv/bin/activate
avril run "TOI-786" --backend cpu
```

CUDA 12 Linux/WSL2：

```bash
./setup_avril.sh cuda
source .venv/bin/activate
avril doctor --backend cuda
avril run "TOI-786" --backend cuda
```

默认会完成官方 SPOC 2 分钟光变下载、完整周期搜索、五个确定性种子投票、事件低点
与谐波归族、精确周期/历元重拟合、逐事件方向检查、4,096 次精确红噪声/相位平移
检验和物理排假。输出的是候选和审计记录，不是行星确认，也不是行星概率。

V3 的加速不是截峰或缩周期网格。它先跑确定性的事件一致性、目录/谐波/缺口漏斗、物理和参考排假；已经
不可能成为自动高优先级的候选仍完整保留，但不再浪费 4,096 次零假设计算。谐波只
允许“直接低点证据 + 胜方自身通过精确零假设”触发跳过；胜方失败时败方自动回退精算。
只有 2–3 次事件的长周期信号保留为人工复核，自动高优先级至少需要 4 次支持事件。

Avril 同时提供与 SPOC 明确分开的 FFI 流程：下载目标全部可用的 TESSCut 像素，做
背景扣除和 1.5/2.0/2.5/3.0 像素多口径自测光，再默认使用最新优化的 Avril Core 全流程。
`--source auto` 会先找完整 SPOC，少于三个官方 sector 才回退 FFI。像素定位默认不跑；
`--pixels followup` 只对已通过红噪声和物理排假的高优先候选运行。FFI 原始像素是
生成光变的必要输入，不等于后置像素定位。
只有实际为 3–4 个 sector 时才可显式选择 `Avril Pro`；Pro 仍是同一套新
Avril 管线，仅更保守地拆分长时间基线上会漂移的邻近周期，不调用旧多视图代码。

```bash
avril run "TOI-786" --source spoc --pixels none
avril run "TIC 123456789" --source ffi
avril run "TIC 123456789" --source ffi --ffi-engine pro
avril run "TIC 123456789" --source auto --pixels followup
```

已知目标回归：L98-59 约 4 分 14 秒并只把三颗已知行星列为自动高优先级；TOI-786
约 3 分 34 秒并找回 38.553883 天长周期；TOI-125 的三颗已知行星排第 1–3；
TOI-1269 的两颗已知行星排第 1–2。计时不含网络下载和可选像素复核。

算法压缩包与网页压缩包完全分开。详细安装、性能调优、踩坑记录和科学边界见
[`docs/`](docs/)。
