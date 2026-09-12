# -*- coding: utf-8 -*-
"""Q3 效率诊断: 固定成本 / 边际成本 分解 + 时间去向 + 理论下界."""
import os, json, math, glob
import numpy as np
import pandas as pd

D = r'E:\Draft\MathModel\cmc26\logs_q3_batch\stats_csv'
runs = pd.read_csv(os.path.join(D, '每轮明细.csv'))
ops = pd.read_csv(os.path.join(D, '操作分解.csv'))
ph = pd.read_csv(os.path.join(D, '阶段分解.csv'))
ch = pd.read_csv(os.path.join(D, '每频道明细.csv'))

ARENA_R = 1800.0
SPEED = 5.0

print('=' * 72)
print('1. 时间去向 (10 轮平均, 秒 / 占比)')
print('=' * 72)
piv = ops.pivot_table(index='category', values='seconds', aggfunc='mean').sort_values('seconds', ascending=False)
tot = piv['seconds'].sum()
for k, v in piv['seconds'].items():
    print(f'  {k:<12s} {v:8.1f} s   {v/tot*100:5.1f}%')
print(f'  {"合计":<12s} {tot:8.1f} s')

print()
print('=' * 72)
print('2. 阶段分解 (10 轮平均)')
print('=' * 72)
pv = ph.pivot_table(index='phase', values='seconds', aggfunc='mean')
for k, v in pv['seconds'].items():
    print(f'  {k:<22s} {v:8.1f} s   {v/pv["seconds"].sum()*100:5.1f}%')

print()
print('=' * 72)
print('3. 固定成本 / 边际成本 回归:  vt = a + b * n_cleared')
print('=' * 72)
n = runs['cleared'].to_numpy(float)
vt = runs['vt_total_s'].to_numpy(float)
b, a = np.polyfit(n, vt, 1)
pred = a + b * n
r2 = 1 - ((vt - pred) ** 2).sum() / ((vt - vt.mean()) ** 2).sum()
print(f'  vt = {a:.0f} + {b:.1f} * n     R^2 = {r2:.3f}')
print(f'  => 固定成本 a = {a:.0f} s,  边际成本 b = {b:.0f} s/源')
print(f'  => T_avg(n) = {a:.0f}/n + {b:.0f}')
for k in (10, 12, 12.5, 14, 16, 20):
    print(f'     n={k:<5} 预测 T_avg = {a/k + b:6.1f} s/个')
print(f'  实测 T_avg(12.5) = {vt.mean()/n.mean():.1f} s/个')
print(f'  固定成本占当前总时间: {a/vt.mean()*100:.1f}%')

print()
print('=' * 72)
print('4. 移动效率: 实际路径 vs 理论最短巡回')
print('=' * 72)
A = math.pi * ARENA_R ** 2
for _, r in runs.iterrows():
    nn = r['cleared']
    # BHH 渐近式: L ~ 0.7124 * sqrt(n * A) (欧氏 TSP, 圆盘内均匀点)
    L_tsp = 0.7124 * math.sqrt(nn * A)
    # 加上从原点出发并返回? 策略不要求返回, 只算开放路径
    print(f"  r{int(r['run']):02d}: n={int(nn):2d}  实测移动 {r['move_distance_m']:7.0f} m  "
          f"TSP下界 {L_tsp:6.0f} m  比值 {r['move_distance_m']/L_tsp:4.2f}")
ratio = (runs['move_distance_m'] / (0.7124 * np.sqrt(n * A)))
print(f'  平均比值 = {ratio.mean():.2f}  (1.0 = 理论最优巡回)')

print()
print('=' * 72)
print('5. 单源动作开销')
print('=' * 72)
print(f"  测向次数/源      : {(runs['n_measure']/n).mean():.1f}")
print(f"  换频次数/源      : {(runs['n_switch']/n).mean():.1f}")
print(f"  清除尝试次数/源  : {(runs['n_clear']/n).mean():.2f}")
print(f"  清除失败次数/轮  : {runs['n_clear_fail'].mean():.1f}  "
      f"(浪费 {runs['n_clear_fail'].mean()*3:.0f} s + 移动)")
print(f"  方位测量数/源    : {ch[ch.detected==1].groupby('run')['n_bearings'].sum().mean()/n.mean():.2f}")
lc = ch[ch.locate_clear_time_s.notna()]['locate_clear_time_s']
print(f"  单源 检测->清除   : 均值 {lc.mean():.0f} s, 中位 {lc.median():.0f} s, "
      f"最大 {lc.max():.0f} s")

print()
print('=' * 72)
print('6. 理论下界粗估 (n=12.5)')
print('=' * 72)
n0 = n.mean()
L_tsp = 0.7124 * math.sqrt(n0 * A)
t_move_floor = L_tsp / SPEED
t_init = 119.0                      # 原点 20 频道: 20*5 + 19*1
t_locate = n0 * 2 * 6.0             # 每源至少 2 次交会测向 = 2*(5+1)
t_clear = n0 * 5.0                  # 每源 1 次成功清除
floor = t_init + t_move_floor + t_locate + t_clear
print(f'  初始扫描(硬)      {t_init:7.0f} s')
print(f'  移动地板(TSP)     {t_move_floor:7.0f} s   ({L_tsp:.0f} m / {SPEED} m/s)')
print(f'  定位测量(2次/源)  {t_locate:7.0f} s')
print(f'  清除(1次/源)      {t_clear:7.0f} s')
print(f'  ----------------------------------')
print(f'  下界 vt           {floor:7.0f} s  -> T_avg = {floor/n0:.0f} s/个')
print(f'  实测 vt           {vt.mean():7.0f} s  -> T_avg = {vt.mean()/n0:.0f} s/个')
print(f'  效率 = 下界/实测 = {floor/vt.mean()*100:.0f}%   (潜在提升空间 {vt.mean()/floor:.2f}x)')
