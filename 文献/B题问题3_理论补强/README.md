# B题问题 3 理论补强文献包（2026 国赛：7 个理论缺口的定向检索）

> **目的**：针对问题 3/4 求解方案中"没有理论支撑、只能靠数值实验调"的 7 个模块，定向检索文献并提炼为可直接引用的定理/公式/算法保证。每份提炼文档自包含：书目信息、逐字定理提取、B 题映射（含不适配处）、GB/T 7714 引用条目。
> **产出时间**：2026-09-11。全文存档 31 篇见 `fulltext/`（开放获取；来源 URL 见各文件头部）。
> **供问题 3/4 建模会话使用**：按下方"缺口→文档"映射直接取用，无需重复检索。

## 缺口 → 文档映射（按建模价值排序）

| 优先级 | 理论缺口（现状） | 文档 | 一句话落点 |
|---|---|---|---|
| ★★★ | **1. 探索-收割切换时机**（δ_detour 阈值与 w_e/w_c/λ 纯调参） | `文献提炼_07_探索收割切换_在线搜索与竞争比.md` | δ_detour → **SmartStart 阈值** L≤(Θ−1)t（Θ=2，竞争比 2.94/精确 2）；w_e/w_c → **单旋钮 α**（cr=1+2α²/(α−1) 显式权衡）；收割= 阶段式批量 4-竞争 + 环形序不跳跃；反轮询论证（m 射线加性项、d^{3/2} 下界） |
| ★★★ | **2. 负信息选点理论保证**（面积贪心无保证） | `文献提炼_08_自适应次模性_负信息贪心理论界.md` | 面积贪心 = AdaptiveGreedy：G&K Def 3.2/3.3 + Thm 5.2（1−1/e）/5.8（ln 型）；证明模板 = Javdani L1–3（只交不并/固定动作固定剪除/自动 self-certifying）；无先验版 = Yuan-Tang；失效兜底 = γ 比率；**一停点多频道 = Chen-Krause 批式** |
| ★★★ | **5. 多目标测点联合选址**（sinγ 加权无权值依据） | `文献提炼_11_多目标选址与集员_几何分数.md` | sinγ 加权 → **集员 E/D 分数**（Π_k det S_{I,k}，距 oracle 4.1%），**权=各源候选域直径**（第一次可推导）；≥3 源无同时最优（Moreno 已证）→ UTMOST 求 tradeoff；2 源特例可写小引理；距离修正 1/d² |
| ★★ | **3. 单停点测量组合**（6 s/项、预算选组） | `文献提炼_09_次模测量调度_预算约束选测.md` | 决策树：按个数=uniform matroid（**1−(1−1/k)^k，k=1 时贪心即最优**）；按秒=knapsack（(1−1/e)OPT−2Lε/c_min）；实现=CELF lazy greedy；次模性清单（检测收益=覆盖自动次模；面积收缩需走提炼 08 的 Javdani 路线）；⚠ Krause-Guestrin 2005 出处更正为 CMU 技术报告 |
| ★★ | **4. 贪心 vs lookahead**（rollout 增益未知） | `文献提炼_10_rollout与MCTS_决策改进.md` | **做不会亏**（Bertsekas Prop 2.1 逐状态不劣于基策略，定理非经验）；代价上限秒级（Rückin：CMA-ES/MCTS 5–6 s/次 vs 一次停点 5+ s）；动作空间需裁到 ~10²；性能差需自测 A/B |
| ★ | **6. 空频道序贯检验**（攒不存在证据） | `文献提炼_12_空频道序贯检验_群检测与搜索论.md` | 次数下界 = log₂(候选假设数)（Aldridge 计数界）；策略模板 = **二分候选区域**（Hwang 最优性）；逐次记账 = Stone (4.1) 位置依赖 p_d 更新（停点在候选域外=零证据）；**"位置依赖测试效力群检测"为理论空白，可声明自建贡献** |
| ★ | **7. 定向源朝向反演**（问题 4 预研） | `文献提炼_13_定向源朝向反演_几何约束传播.md` | 扇形约束传播 = Desrochers-Jaulin Minkowski/separator（收敛+ε 逼近免费）；可辨识性 = Williams CRLB/FIM + **主瓣边缘选点**（朝向最敏感处在候选扇形边界附近）；最少测点数无现成理论 → 自建引理方向（log₂ 朝向假设数 + Hwang 型二分，与提炼 12 同构） |

## 使用顺序建议

1. **先读 08**（用户指定最值钱）：建模时把"面积收缩贪心"装进 AdaptiveGreedy 框架，引理验证清单在 §3；
2. **再读 07**：把 δ_detour/权重换成 SmartStart 阈值 + α 旋钮（论文里"策略结构"一节的定理化）；
3. **再读 11**：把所有 sinγ 加权评分换成 E/D 分数（一处改动，全局受益）；
4. 09/10 为算法改进层（演练测试可验证）；12/13 按需（12 的"理论空白声明"本身是加分项；13 供问题 4 会话开题）。

## 去重说明（与既有文献包的关系）

- **不重复**：`B题问题2/文献提炼_01`（单源 FIM/90°）、`B题问题3/文献提炼_02–06`（CPP 覆盖、Xiao 综述、RFS-PHD/事件触发、Mohammadi）——本包 7 个缺口全部为上述未覆盖领域；
- **交叉引用而非重复展开**：Sahu 式 26 与问题 2 的 FIM 同构（交叉验证）；Rückin 为 Xiao 综述中"8–10×"的原文出处；Moreno-Salinas 2013 即 `B题问题2` README 待读队列头号（现已精读，注意其是 range-only）；
- **已核实版本更正**（引用前必查）：Golovin-Krause 2011 = **JAIR** 42:427–486（非 JMLR）；Krause-Guestrin 2005 预算化 = **CMU-CALD-05-103 技术报告**（非 IPDPS）；Chen-Krause 2013 标题为 *Near-optimal Batch Mode Active Learning...*。

## 全文存档清单（fulltext/，31 文件）

| 提炼文档 | 对应全文 |
|---|---|
| 07 | birx2019、demaine2004、kaminski2026、angelopoulos2018、bienkowski2021、bansal2022、langetepe2025、lyu2026 |
| 08 | golovin2011_jair（正式版）、golovin2011（arXiv v5）、chenkrause2013(+_supp)、javdani2012、fujii2019、yuantang2022 |
| 09 | krause2008、liu2020survey、singh2009 |
| 10 | bertsekas2022、bertsekas2020multiagent、rueckin2021 |
| 11 | moreno2013(.txt/.xml)、calafiore2026range、calafiore2026anchor、sahu2021、dogancay2010、jauberthie2018 |
| 12 | aldridge2019（第二版）、stone2014、debonis2023、（附：davey2016） |
| 13 | desrochers2017、williams2024、joneidi2019 |

（.pdf 原件：calafiore2026anchor/range、krause2008、liu2020survey、sahu2021、singh2009、moreno2013.xml。）

## 检索环境备忘（供后续会话，勿再踩坑）

- **可用**：arXiv（含 export API）、OpenAlex、PMC/EuropePMC、PMLR、JMLR/JAIR、MIT 作者主页、Springer link、Project Euclid、Zenodo、HAL；
- **不可用**：IEEE Xplore（418 反爬）、ScienceDirect/Elsevier（反爬）、Semantic Scholar（429 限流）、archive.org 及 web.archive.org、DTIC、NPS Calhoun、figshare/KiltHub（403）、DuckDuckGo、Bing（出口被本地化，学术结果被过滤）、DBLP（bot 防护）；
- **付费墙仅摘要已标注**：Sviridenko 2004、Koopman 1957、Stone 1975 专著、Jaillet-Wagner 两篇、Kao-Reif-Tate 1996、Garulli-Vicino 2001、Gu-He-Han 2015、Cheng 2023——引用须标"仅摘要/二手引用"。

## 已确认的理论空白（论文可作为自建贡献声明）

1. **位置依赖测试效力的群检测**（提炼 12 §5）：无现成文献——"在哪测、测哪组"用 Aldridge 计数界 + Stone p_d 模型组合自建；
2. **半平面带交集确定朝向的最少测点数**（提炼 13 §4）：无现成闭式——自建引理方向：log₂(朝向假设数) 下界 + Hwang 型二分策略；
3. **多目标 bearing 一站解析冲突**（提炼 11）：开放文献无解析结论（Moreno 仅证 2 源可同时最优、≥3 源需 tradeoff 且权值 mission-dependent）——本包的 E/D 分数方案即填补此空白的自建贡献。
