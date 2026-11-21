# 补充材料：熵产生的三层上界的数值核验

本仓库包含重现论文 §6、附录 A 与 §8 中全部数值结果所需的脚本与输出。

- 仓库地址：<https://github.com/KaifaChen/entropy>

## 环境

- Python 3（任一版本均可，无版本特定语法）
- NumPy（唯一依赖）
- 运行：`python -u <script>.py`
- 所有随机数种子固定（`20261121`），故结果可逐位复现
- 输出以 UTF-8 存于 `outputs/<script>.txt`

## 脚本与论文的对应

| 脚本 | 对应的论文位置 | 产出 |
| :--- | :--- | :--- |
| `lemma_check.py` | 附录 A（引理 1、引理 2）；§6.1、§6.2 | 引理 1：$4\times10^5$ 组，0 违例，LHS/RHS 最小值 $1.000000000047$；引理 2：$n\in\{2,3,5,10\}$ 各 $2\times10^4$ 组加极端单点分布，0 违例，$JS/H^2$ 最大值 $1.000000$ |
| `verify_level25.py` | §4.2、§6.5、附录 A | 9 条链上 $\sum_{i<j}I_{\rm edge}(0)=H(\pi)$（最大偏差 $5.55\times10^{-17}$）且 $\sum I_{\rm edge}(j_{\rm ss})=0$（最大 $8.04\times10^{-17}$） |
| `container_phi4.py` | §6.3、§6.4 | 环上 $\phi^4$ 场离散化所得的 8 状态马氏链，6 个驱动值 $f=0,\ 0.25,\ 0.5,\ 1,\ 2,\ 4$；逐例检查三层不等式 |
| `theorem_cases.py` | §8.2（F1、F2）、§6.3 | 均匀环 $m=3,4,5$ × $r=1.5,3,10$（9 例）与非均匀势环（6 例）上 $4JS$ 与 TUR 的比较，共 15 例 |
| `tur_figure_eight.py` | §8.2（F3） | figure-eight 网络（$V=5$、$E=6$、$b_1=2$）上 12 组 $(r_A,r_B)$ 的 $4JS$ 与最优 TUR $J^{\mathsf T}D^{-1}J$ 比较 |
| `tur_exact_numbers.py` | §8.2 的精确数字表 | $(r_A,r_B)=(50,1),(10,10),(5,1)$ 的 $\sigma$、$J$、$D_J$、$J^{\mathsf T}D^{-1}J$ 的十位数字 |
| `verify_ratio_r.py` | §7.3、附录 A | 单循环驱动下 $H/\Delta S=\tanh(a/4)/a$ 的逐 $m$ 核验，并给出 $a\to0$ 的 $1/4$ 极限与 $a=0.5$ 处的 $JS/\Delta S$ |

## 约定的细节

- 平稳分布一律由线性方程组 $(\pi T=\pi,\ \sum\pi_i=1)$ 的最小二乘解求得，再截断归一化。
- $\Delta S$、$JS$ 的 KL 求和用 `1e-300` 截断以避免 $\ln0$。
- $H(\pi)$ 按 $i<j$ 求和（这是唯一正确的写法；覆盖全部 $(i,j)$ 会多算一倍，见下文"已知错误"）。
- TUR 的扩散系数取 $D_J\equiv\frac12\mu''(0)$，使 $\mathrm{Var}(J_t)\simeq2D_Jt$；$\mu(z)$ 为倾角生成元的 Perron 根，差分步长 $h$ 在 `theorem_cases.py` 与 `tur_exact_numbers.py` 中为 $10^{-4}$，在 `tur_figure_eight.py` 中为 $10^{-3}$（见下文"须在论文中分注"）。

## 与论文正文的逐项对照

逐项核对的结果如下。凡与论文不符者，均在此列出，并在论文中按本目录的输出更正。

| 论文声称 | 本目录输出 | 结论 |
| :--- | :--- | :--- |
| 引理 1：$4\times10^5$ 组，0 违例，最小比值 $1.000000000047$ | 400000 组，0 违例，$1.000000000047$ | 一致 |
| 引理 2：0 违例，$JS/H^2$ 最大为 1 | 0 违例，$1.000000$ | 一致 |
| §4.2、§6.5：9 条链，差 $\le5.6\times10^{-17}$ | 9 条链，最大 $5.55\times10^{-17}$ | 一致 |
| §8.2：F1 9 例、F2 6 例、F3 12 例，全部 $4JS$ 更紧 | 15 例 + 12 例，全部 $4JS$ 更紧 | 一致 |
| §8.2 精确数字表（三例） | 逐位一致 | 一致 |
| §6.3：$\phi^4$ 场链"12 组参数" | 实为 6 组（$f=0,0.25,0.5,1,2,4$） | 论文已更正为 6 组 |
| §6.4：场链"$0.2481$，与解析 $0.2487$ 差 $0.2\%$" | 场链在 $f=0.5$ 处 $H/\Delta S=0.248711$（即 $r(0.5)$）、$JS/\Delta S=0.247428$；输出中不存在 $0.2481$ | 论文已按本输出改写 |
| §7.3：纯环上"$JS/\Delta S=0.2470$" | $0.247428$ | 论文已更正为 $0.2474$ |
| 附录 A：$r(a)$ 的"$m=3$ 至 $6$、$K=2$ 至 $5$，共 8 种组合" | $\Delta S$、$H$、$JS$ 只依赖 $T$ 与 $\pi$，发射字母表大小不进入，故"与 $K$ 无关"恒真；有内容的只是对 $m$ 的无关性 | 论文已删去关于 $K$ 的表述 |
| 附录 A：TUR 差分步长"$h=10^{-4}$" | F1、F2 与精确数字表用 $10^{-4}$；F3 的 12 例用 $10^{-3}$ | 论文已分注 |

## 已知错误与已排除的版本

本目录只收录经过核对、输出与论文一致的脚本。历史版本中有下列已知错误的脚本，均未收录：

- 早期版本把 $H(\pi)$ 按**全部** $(i,j)$ 求和（而非 $i<j$），导致 $H$ 大一倍，从而产生虚假的"违例"。本目录的 `container_phi4.py` 是修正版。
- 早期版本的 figure-eight 构造只给每条无向边设一个方向，导致转移矩阵的行和不为 1（不是随机矩阵）。本目录的 `tur_figure_eight.py` 是修正版。
- 非一阶过程的保持率 $\rho_n$ 的四个实现互不一致，且未通过"$n=1$ 时应还原标准公式"的自检，故论文 §7.2 不作定量主张，本目录亦不含相关脚本。

## 复现命令

```
python -u lemma_check.py          > outputs/lemma_check.txt
python -u verify_level25.py       > outputs/verify_level25.txt
python -u container_phi4.py       > outputs/container_phi4.txt
python -u theorem_cases.py        > outputs/theorem_cases.txt
python -u tur_figure_eight.py     > outputs/tur_figure_eight.txt
python -u tur_exact_numbers.py    > outputs/tur_exact_numbers.txt
python -u verify_ratio_r.py       > outputs/verify_ratio_r.txt
```
