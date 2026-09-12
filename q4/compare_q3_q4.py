# -*- coding: utf-8 -*-
"""Q3 vs Q4 对比: 关键指标 / 成本模型(a,b) / 时间去向 / 路径效率."""
import os
import math
import numpy as np
import pandas as pd

BASE = r'E:\Draft\MathModel\cmc26'
Q3 = os.path.join(BASE, 'logs_q3_batch', 'stats_csv')
Q4 = os.path.join(BASE, 'logs_q4_batch', 'stats_csv')
ARENA_R, SPEED = 1800.0, 5.0

r3 = pd.read_csv(os.path.join(Q3, '每轮明细.csv'))
r4 = pd.read_csv(os.path.join(Q4, '每轮明细.csv'))
o3 = pd.read_csv(os.path.join(Q3, '操作分解.csv'))
o4 = pd.read_csv(os.path.join(Q4, '操作分解.csv'))


def cost_model(df):
    n = df['cleared'].to_numpy(float)
    vt = df['vt_total_s'].to_numpy(float)
    b, a = np.polyfit(n, vt, 1)
    pred = a + b * n
    r2 = 1 - ((vt - pred) ** 2).sum() / ((vt - vt.mean()) ** 2).sum()
    return a, b, r2


print('=' * 78)
print('1. 关键指标对比 (均值 ± 标准差)')
print('=' * 78)
print(f"{'指标':<22}{'Q3 (全向源)':>22}{'Q4 (混合定向源)':>24}{'Q4/Q3':>10}")
rows = [
    ('清除源数', r3['cleared'], r4['cleared'], ''),
    ('漏检数(检测未清除)', r3['missed'], r4['missed'], ''),
    ('虚拟总时间 s', r3['vt_total_s'], r4['vt_total_s'], ''),
    ('T_avg s/个', r3['avg_per_source_s'], r4['avg_per_source_s'], ''),
    ('测向次数', r3['n_measure'], r4['n_measure'], ''),
    ('移动距离 m', r3['move_distance_m'], r4['move_distance_m'], ''),
    ('清除成功率', r3['clear_success_rate'], r4['clear_success_rate'], ''),
    ('现实耗时 s', r3['real_elapsed_s'], r4['real_elapsed_s'], ''),
]
for name, s3, s4, _ in rows:
    m3, m4 = s3.mean(), s4.mean()
    ratio = f'{m4/m3:6.2f}x' if m3 else ''
    print(f'{name:<22}{m3:>14.2f} ±{s3.std():<7.2f}{m4:>14.2f} ±{s4.std():<9.2f}{ratio:>10}')

print()
print('=' * 78)
print('2. 成本模型  vt = a + b·n      (T_avg = a/n + b)')
print('=' * 78)
a3, b3, r23 = cost_model(r3)
a4, b4, r24 = cost_model(r4)
print(f'  Q3: vt = {a3:6.0f} + {b3:5.0f}·n     R2={r23:.3f}   固定成本占比 {a3/r3["vt_total_s"].mean()*100:.0f}%')
print(f'  Q4: vt = {a4:6.0f} + {b4:5.0f}·n     R2={r24:.3f}   固定成本占比 {a4/r4["vt_total_s"].mean()*100:.0f}%')
print()
print(f'  固定成本 a: Q4 是 Q3 的 {a4/a3:.2f} 倍   (判空证书 7点 → 26点 的代价)')
print(f'  边际成本 b: Q4 是 Q3 的 {b4/b3:.2f} 倍')
print()
print('  T_avg 随源数变化:')
print(f'    {"n":<6}{"Q3":>10}{"Q4":>10}{"差距":>10}')
for k in (10, 12, 13, 15, 16):
    t3, t4 = a3 / k + b3, a4 / k + b4
    print(f'    {k:<6}{t3:>10.0f}{t4:>10.0f}{t4/t3:>9.2f}x')

print()
print('=' * 78)
print('3. 时间去向对比 (秒 / 占本问总时间)')
print('=' * 78)
p3 = o3.pivot_table(index='category', values='seconds', aggfunc='mean')
p4 = o4.pivot_table(index='category', values='seconds', aggfunc='mean')
t3, t4 = p3['seconds'].sum(), p4['seconds'].sum()
print(f"{'类别':<14}{'Q3 s':>10}{'Q3 %':>8}{'Q4 s':>10}{'Q4 %':>8}")
for k in p3.index:
    v3 = p3.loc[k, 'seconds']
    v4 = p4.loc[k, 'seconds'] if k in p4.index else 0.0
    print(f'{k:<14}{v3:>10.1f}{v3/t3*100:>7.1f}%{v4:>10.1f}{v4/t4*100:>7.1f}%')

print()
print('=' * 78)
print('4. 路径效率: 实测移动 / TSP 理论最短巡回')
print('=' * 78)
A = math.pi * ARENA_R ** 2
for tag, df in (('Q3', r3), ('Q4', r4)):
    n = df['cleared'].to_numpy(float)
    ratio = df['move_distance_m'].to_numpy(float) / (0.7124 * np.sqrt(n * A))
    print(f'  {tag}: 平均移动 {df["move_distance_m"].mean():7.0f} m, '
          f'TSP下界 {(0.7124*np.sqrt(n*A)).mean():6.0f} m, '
          f'比值 {ratio.mean():.2f}  (最差 {ratio.max():.2f})')

print()
print('=' * 78)
print('5. 结论')
print('=' * 78)
print(f'  Q4 的 T_avg ({r4["avg_per_source_s"].mean():.0f}s) 是 Q3 ({r3["avg_per_source_s"].mean():.0f}s) '
      f'的 {r4["avg_per_source_s"].mean()/r3["avg_per_source_s"].mean():.2f} 倍')
print(f'  主因: 判空固定成本 {a3:.0f}s → {a4:.0f}s ({a4/a3:.1f}倍), 这是 7点覆盖网 → 26点环抱证书的刚性代价')
print(f'  Q4 测向次数 {r4["n_measure"].mean():.0f} vs Q3 {r3["n_measure"].mean():.0f} '
      f'({r4["n_measure"].mean()/r3["n_measure"].mean():.1f}倍) —— 定向源盲区导致大量重复探测')
miss4 = int(r4['missed'].sum())
print(f'  Q3 漏检 {int(r3["missed"].sum())} 个, Q4 漏检 {miss4} 个'
      + ('  <-- Q4 需关注' if miss4 else ''))
