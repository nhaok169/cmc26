# 文献提炼 06：Mohammadi & Shafer 2025 —— 多目标精确定位的 UAV 航路点智能选择【自行总结】

> ⚠️ **诚实声明**：本文全文仅存于 IEEE Xplore，经多途径尝试均被拦截（IEEE 418 反爬、Unpaywall/OpenAlex 无副本、NAU Pure 门户只挂 DOI 链接、作者无预印本、DuckDuckGo/Bing 检索无第三方副本、Semantic Scholar 引用无语境）。**以下内容由会话基于官方摘要 + 检索要点 + 同方法族的全文文献（Wang 2023 / Xiao 2026 / Doğançay-Hmam 2024，均全文精读）自行总结**，方法细节处已标注推断依据，引用时请注意核实。
> **相关度**：★★★★（与 B题问题 3 设定同构性最强的工程文献）

---

## 1. 文献信息

| 项 | 内容 |
|---|---|
| 标题 | UAV Path Planning for Precision Multi-Target Localization |
| 作者 | Mahsa Mohammadi, Michael W. Shafer（Northern Arizona University，机械工程系） |
| 来源 | IEEE Access 2025, 13: 63715-63728（14 页，开放获取但 IEEE 反爬） |
| DOI | 10.1109/ACCESS.2025.3558700 |
| 摘要全文 | 已获取（存档 `s2_mohammadi.json`） |

## 2. 摘要原文要点（可靠层：直接来自官方摘要）

场景：单架 UAV 携带测向设备定位**多个静止的 VHF 无线电标签野生动物**。核心问题：**航路点的智能选择**。逐句拆解：

1. "At each designated waypoint, the UAV obtains **bearing measurements** to tagged animals, **considering the associated uncertainty**" —— 停点测向、量测含不确定度（与 B题示向度 ±1° 同构）
2. "intelligently recommends **subsequent locations that minimize predicted localization uncertainty**" —— 下一位置选择准则 = **最小化预测定位不确定度**（预测型目标）
3. 约束三件套："**mission time**（任务时间最短）、**keeping the UAV within signal range**（保持在信号有效范围内）、**maintaining a suitable distance from targets**（与目标保持适当距离，避免惊扰野生动物）"
4. 评估："**uncertainty reduction throughout the mission**（任务全程不确定度收缩曲线）、与 ground truth 对比、**mission time 蒙特卡洛分析**"

## 3. 方法族还原（推断层：基于同族全文文献的可靠推断）

"最小化预测定位不确定度"的航路点选择 = **A-最优 FIM 预测框架**的工程实例，与已全文精读的三篇构成同一方法族，可高度置信地还原其骨架：

| 摘要表述 | 方法族对应（全文依据） | B题映射 |
|---|---|---|
| bearing measurements + uncertainty | 测向 FIM：信息只在垂直视线方向、随距离平方衰减（Wang 2023 式10；Xiao 2026 §3.1） | 示向度 ±1°，σ=0.577° |
| minimize predicted uncertainty | 候选下一停点的 $\operatorname{tr}(J^{-1})$（A-最优，误差椭圆半轴平方和）或 GDOP 预测值最小——**多目标时对各目标的不确定度求和/加权**（推断） | tr(J⁻¹)=σ²(r₁²+r₂²)/sin²θ₁₂（问题2结论），候选点逐一算分 |
| within signal range | 检测点须在源接收半径内的硬约束（B题 1000–1500m；论文对应 VHF 信号范围） | 同 |
| suitable distance | 最小距离约束（避免过近，B题对应 ≤5m 信号过强无示向度） | 同 |
| Monte Carlo mission time | 随机化源位置/量测噪声的批量仿真评估（B题模拟器演练测试同构） | 演练测试统计量 |

**判断依据**：该文 2025 年发表、引用 Wang/Bishop 一系的 FIM 几何文献是此领域标准做法（Xiao 2026 引言明确该领域 "conventional FIM-based optimization typically maximizes the determinant"）；"predicted localization uncertainty" 这一措辞即 FIM/CRLB 预测语义。其贡献应在于**多目标场景下的航路点选择启发式/算法与工程约束的整合**，而非新的定位理论。

## 4. 对 B题问题 3 的用法建议

1. **作为"同构案例"引用**（引言/模型 motivation）：单平台、停点测向、多静止源、信号半径约束、时间最短——五要素与 B题逐条对应，证明该问题设定有成熟工程先例
2. **策略借鉴**：逐停点"预测各源不确定度 → 选最小化总分+时间代价的下一停点"的闭环，正是文献提炼04 §4 Phase A/B 的在线决策核心；本文可作为该策略的独立佐证
3. **评估方法借鉴**：任务全程不确定度收缩曲线 + 蒙特卡洛任务时间分析 → B题论文的"演练测试"图表设计（每个源定位区域直径/面积随停点数收缩的曲线；平均定位清除时间的分布箱线图）
4. **不建议**引用其具体公式（无法核实全文细节）；引用时以摘要级表述为宜

## 5. 引用（GB/T 7714-2015）

```
[?] MOHAMMADI M, SHAFER M W. UAV path planning for precision multi-target localization[J]. IEEE Access, 2025, 13: 63715-63728. DOI:10.1109/ACCESS.2025.3558700.
```

## 6. 若需全文的获取途径（留给后续会话）

- 校园网/机构订阅下直接访问 IEEE Xplore: https://ieeexplore.ieee.org/document/10955259/
- 通信作者（Michael W. Shafer，NAU 机械工程系）邮件索取作者版 —— 学术惯例可行
- 注意：本环境 IP（59.64.129.114）已被 IEEE 标记 418，换网络环境可能直接解除
