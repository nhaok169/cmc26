# 2026 高教社杯全国大学生数学建模竞赛 B 题

无线电干扰源的快速自动定位与清除。

## 成品

| 文件 | 说明 |
|---|---|
| `paper/无线电干扰源的快速自动定位与清除.pdf` | 终稿 PDF |
| `paper/无线电干扰源的快速自动定位与清除.docx` | 终稿 Word |
| `paper/B题_全文.md` | 合稿源文 |
| `submit/准成稿/` | 论文 PDF + `支撑材料.zip` |

由源文生成 Word：`python paper/build_docx.py`

## 代码

| 问 | 入口 | 说明 |
|---|---|---|
| 一 | `q1/solver.py` | 可行域分类与作图 |
| 二 | `q2/geometry.py` | 交会几何、第二检测点 |
| 三 | `python q3/run_q3.py` | 原点扫频 + 1150 m 六点环 + 追击 |
| 四 | `python q4/run_q4.py` | 26 点环抱证书 + 合并巡回（`ledger2.py`） |

问题三 / 四连模拟器时用环境变量 `ROBOT_ID` 设置队号，或 `python q3/run_q3.py --watch` / `python q4/run_q4.py --watch`。

## 正式测试

- 问题三：三局 100% 清除，均值 **285.34 s/个**
- 问题四：三局 100% 清除，均值 **600.26 s/个**

## 其它

- `contest/` 赛题与附件
- `simulator/` 官方模拟器
- `AI工具使用详情.pdf` 竞赛要求的 AI 使用说明
