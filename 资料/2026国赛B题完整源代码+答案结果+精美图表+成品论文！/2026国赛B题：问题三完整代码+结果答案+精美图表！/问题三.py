# -*- coding: utf-8 -*-
"""
问题三：全向干扰源的机器狗自动定位与清除策略
1) 自建与官方协议等价的模拟器（全向源，10~16 个，频道互异，有效半径 1000~1500 m）
2) 策略：环形锚点覆盖扫描(7锚点保证任意源距某锚点≤1000m) → 示向度交会定位(最小二乘)
         → 近邻TSP清除路径 + 就近寻的(homing) → 验证
3) 演练测试(20次) + 3 次正式测试，输出统计与行为日志
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import *
import itertools, math

FIG = FIG_DIR
OUT = OUT_DIR
ARENA_R = 1800.0
V = 5.0            # 速度 m/s
T_DETECT = 5.0     # 检测耗时
T_SWITCH = 1.0     # 频道切换
T_CLEAR_OK = 5.0   # 精确定位+清除
T_CLEAR_FAIL = 3.0 # 精确定位未发现
CLEAR_R = 20.0
NEAR_R = 5.0

def unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])

def left_normal(deg):
    a = np.deg2rad(deg)
    return np.array([-np.sin(a), np.cos(a)])

def ang_of(v):
    return (np.degrees(np.arctan2(v[1], v[0]))) % 360.0

# ---------------- 模拟器 ----------------
class Simulator:
    """等价于官方 HTTP 接口的本地模拟器（全向源）"""
    def __init__(self, sources, seed=0):
        # sources: dict ch -> dict(pos=array, R=float)
        self.sources = sources
        self.pos = np.array([0.0, 0.0])
        self.channel = 1
        self.time = 0.0
        self.cleared = set()
        self.rng = np.random.default_rng(seed)
        # 分项耗时统计
        self.t_move = 0.0; self.t_switch = 0.0; self.t_detect = 0.0; self.t_clear = 0.0
        self.n_detect = 0; self.n_clear = 0; self.n_switch = 0
        self.log = []

    def _move(self, p):
        p = np.asarray(p, float)
        d = np.linalg.norm(p - self.pos)
        t = d / V
        self.time += t; self.t_move += t
        self.pos = p

    def measure(self, x, y, ch):
        p = np.array([x, y], float)
        self._move(p)
        if ch != self.channel:
            self.time += T_SWITCH; self.t_switch += T_SWITCH; self.n_switch += 1
            self.channel = ch
        self.time += T_DETECT; self.t_detect += T_DETECT; self.n_detect += 1
        src = self.sources.get(ch)
        if src is None or ch in self.cleared:
            res = 'no_signal'
        else:
            G = src['pos']; R = src['R']
            d = np.linalg.norm(p - G)
            if d <= R:
                if d <= NEAR_R:
                    res = 'near'
                else:
                    tb = ang_of(G - p)
                    err = self.rng.uniform(-1.0, 1.0)
                    res = ('direction', round((tb + err) % 360.0, 2))
            else:
                res = 'no_signal'
        self.log.append(('measure', (x, y), ch, res, round(self.time, 3)))
        return res

    def clear(self, x, y, ch):
        p = np.array([x, y], float)
        self._move(p)
        src = self.sources.get(ch)
        if src is not None and ch not in self.cleared and np.linalg.norm(p - src['pos']) <= CLEAR_R:
            self.cleared.add(ch)
            self.time += T_CLEAR_OK; self.t_clear += T_CLEAR_OK; self.n_clear += 1
            res = 'success'
        else:
            self.time += T_CLEAR_FAIL; self.t_clear += T_CLEAR_FAIL; self.n_clear += 1
            res = 'no_target_in_range'
        self.log.append(('clear', (x, y), ch, res, round(self.time, 3)))
        return res

# ---------------- 案例生成 ----------------
def gen_case(rng, n_src=None, n_channels=20):
    if n_src is None:
        n_src = int(rng.integers(10, 17))
    chs = rng.choice(np.arange(1, n_channels + 1), size=n_src, replace=False)
    sources = {}
    for ch in chs:
        ang = rng.uniform(0, 2 * np.pi)
        rr = ARENA_R * np.sqrt(rng.uniform(0, 1))
        pos = np.array([rr * np.cos(ang), rr * np.sin(ang)])
        R = rng.uniform(1000.0, 1500.0)
        sources[int(ch)] = {'pos': pos, 'R': R}
    return sources

# ---------------- 锚点 ----------------
def anchors(center=True, rho=1150.0, n_ring=6):
    A = []
    if center:
        A.append(np.array([0.0, 0.0]))
    for k in range(n_ring):
        th = np.deg2rad(360.0 * k / n_ring)
        A.append(rho * np.array([np.cos(th), np.sin(th)]))
    return A

def coverage_check(rng, A, n=2000):
    """验证锚点覆盖：随机源位置到最近锚点距离 ≤ 1000 的比例"""
    bad = 0
    for _ in range(n):
        ang = rng.uniform(0, 2*np.pi); rr = ARENA_R*np.sqrt(rng.uniform(0,1))
        G = np.array([rr*np.cos(ang), rr*np.sin(ang)])
        if min(np.linalg.norm(G - a) for a in A) > 1000.0:
            bad += 1
    return 1 - bad / n

def localize(bearings):
    """最小二乘交会定位（示向线交点）"""
    A = np.zeros((2, 2)); b = np.zeros(2)
    for P, th in bearings:
        n = left_normal(th)
        A += np.outer(n, n)
        b += (n @ P) * n
    return np.linalg.solve(A, b)

def tsp_order(pts, start):
    """最近邻 TSP 顺序"""
    n = len(pts)
    order = []; cur = np.asarray(start, float)
    rem = set(range(n))
    while rem:
        j = min(rem, key=lambda k: np.linalg.norm(pts[k] - cur))
        order.append(j); cur = pts[j]; rem.remove(j)
    return order

def grid_clear(env, ch, pos, rings=((0.0, 1), (20.0, 6), (40.0, 10), (60.0, 14))):
    """局部网格清除兜底"""
    for r, n in rings:
        for k in range(n):
            th = 2*np.pi*k/max(n, 1)
            p = pos + r*np.array([np.cos(th), np.sin(th)])
            if env.clear(p[0], p[1], ch) == 'success':
                return True
    return False

def home_bisect(env, ch, P, th):
    """单示向度源：源在 P+t·u(th) 上，t∈[5,1000]，沿示向线二分定位并清除"""
    u = unit(th)
    lo, hi = 5.0, 1000.0
    for _ in range(13):
        t = (lo + hi) / 2
        pos = P + t * u
        r = env.measure(pos[0], pos[1], ch)
        if r == 'near':
            return env.clear(pos[0], pos[1], ch) == 'success'
        if r == 'no_signal':
            hi = t
            continue
        diff = abs(((r[1] - th + 180) % 360) - 180)
        if diff < 90:
            lo = t          # 源在更远处
        else:
            hi = t          # 已越过源
    pos = P + ((lo + hi) / 2) * u
    return grid_clear(env, ch, pos)

def clear_target(env, ch, est):
    """移动到估计位置，就近寻的并清除"""
    pos = np.asarray(est, float)
    r = env.clear(pos[0], pos[1], ch)
    if r == 'success':
        return True
    m = env.measure(pos[0], pos[1], ch)
    if m == 'near':
        return env.clear(pos[0], pos[1], ch) == 'success'
    if m != 'no_signal' and m[0] == 'direction':
        for _ in range(14):
            pos = pos + 15.0 * unit(m[1])
            r = env.clear(pos[0], pos[1], ch)
            if r == 'success':
                return True
            m = env.measure(pos[0], pos[1], ch)
            if m == 'near':
                return env.clear(pos[0], pos[1], ch) == 'success'
            if m == 'no_signal':
                return grid_clear(env, ch, pos)
            # direction -> 继续
    return grid_clear(env, ch, pos)

# ---------------- 策略 ----------------
def run_strategy(env, seed, A=None):
    """返回 (cleared_count, time, detail)"""
    if A is None:
        A = anchors()
    active = set(range(1, 21))
    bearings = {ch: [] for ch in range(1, 21)}
    estimates = {}
    # Phase 1 覆盖扫描
    for a in A:
        for ch in sorted(active):
            r = env.measure(a[0], a[1], ch)
            if r == 'near':
                if env.clear(a[0], a[1], ch) == 'success':
                    active.discard(ch)
            elif r != 'no_signal' and r[0] == 'direction':
                bearings[ch].append((a.copy(), r[1]))
                if len(bearings[ch]) >= 2:
                    estimates[ch] = localize(bearings[ch])
                    active.discard(ch)
        if not active:
            break
    # Phase 1.5 单示向度源：沿示向线二分定位清除
    for ch in list(active):
        if len(bearings[ch]) == 1:
            P, th = bearings[ch][0]
            if home_bisect(env, ch, P, th):
                active.discard(ch)
    # Phase 2 清除路径
    cleared_order = []
    if estimates:
        keys = list(estimates.keys())
        pts = [estimates[k] for k in keys]
        for j in tsp_order(pts, env.pos):
            ch = keys[j]
            ok = clear_target(env, ch, estimates[ch])
            if ok:
                cleared_order.append(ch)
                if ch in active:
                    active.discard(ch)
    # Phase 3 验证：中心点复扫所有未确认频道
    for ch in sorted(range(1, 21)):
        if ch in env.cleared:
            continue
        r = env.measure(0.0, 0.0, ch)
        if r == 'near':
            env.clear(0.0, 0.0, ch)
        elif r != 'no_signal' and r[0] == 'direction':
            # 理论不应发生；补清
            P = np.array([0.0, 0.0])
            p2 = P + 400.0*unit(r[1])
            m2 = env.measure(p2[0], p2[1], ch)
            if m2 != 'no_signal':
                if m2 == 'near':
                    env.clear(p2[0], p2[1], ch)
                else:
                    est = localize([(P, r[1]), (p2, m2[1])])
                    clear_target(env, ch, est)
    return len(env.cleared), env.time, dict(
        t_move=env.t_move, t_switch=env.t_switch, t_detect=env.t_detect, t_clear=env.t_clear)

# ---------------- 运行测试 ----------------
def one_trial(seed):
    rng = np.random.default_rng(seed)
    sources = gen_case(rng)
    env = Simulator(sources, seed=seed + 1000)
    n_cleared, t, detail = run_strategy(env, seed + 2000)
    return dict(n_true=len(sources), n_cleared=n_cleared, time=t, **detail)

# ---------------- 图 ----------------
def fig1_trajectory(case, seed):
    """干扰源分布 + 锚点 + 机器狗轨迹"""
    rng = np.random.default_rng(seed)
    sources = gen_case(rng) if case is None else case
    env = Simulator(sources, seed=seed + 1000)
    n_cleared, t, detail = run_strategy(env, seed + 2000)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_aspect('equal', adjustable='box')
    # 目标区域
    ax.add_patch(Circle((0, 0), ARENA_R, fill=False, ec='#94a3b8', lw=1.0, ls='--'))
    # 锚点
    A = anchors()
    ax.plot([a[0] for a in A], [a[1] for a in A], 's', ms=7, color='#f6ad55', mec='k', mew=0.8, zorder=4)
    # 干扰源
    for ch, s in sources.items():
        c = '#2f855a' if ch in env.cleared else '#e53e3e'
        ax.plot([s['pos'][0]], [s['pos'][1]], 'o', ms=9, color=c, mec='k', mew=0.9, zorder=5)
        ax.add_patch(Circle(s['pos'], s['R'], fill=False, ec='#a0aec0', lw=0.5, alpha=0.55))
    # 轨迹
    xs = [0.0]; ys = [0.0]
    for act in env.log:
        xs.append(act[1][0]); ys.append(act[1][1])
    ax.plot(xs, ys, color='#2b6cb0', lw=1.2, alpha=0.9, zorder=3)
    ax.plot([0], [0], '^', ms=10, color='#2b6cb0', mec='k', mew=1.0, zorder=6)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(-1900, 1900); ax.set_ylim(-1900, 1900)
    despine(ax)
    save_fig(fig, '图3-1_干扰源分布与机器狗轨迹')

def fig2_loc_error():
    """定位误差 vs 示向度数量（Monte Carlo）"""
    rng = np.random.default_rng(3)
    data = {2: [], 3: [], 4: []}
    for _ in range(400):
        G = rng.uniform(-1200, 1200, 2)
        m = 4
        pts = rng.uniform(-1500, 1500, (m, 2))
        for k in (2, 3, 4):
            bs = []
            for P in pts[:k]:
                th = ang_of(G - P) + rng.uniform(-1, 1)
                bs.append((P, th))
            est = localize(bs)
            data[k].append(np.linalg.norm(est - G))
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [f'{k}条' for k in (2, 3, 4)]
    bp = safe_boxplot(ax, [data[k] for k in (2, 3, 4)], labels, widths=0.5)
    cols = ['#f6ad55', '#68d391', '#63b3ed']
    for i, c in enumerate(cols):
        bp['boxes'][i].set_facecolor(c)
        bp['medians'][i].set_color('#1a202c')
    ax.set_ylabel('定位误差 (m)')
    ax.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    despine(ax)
    save_fig(fig, '图3-2_定位误差与示向度数量关系')

def fig3_time_composition(trials):
    """时间构成堆叠柱状"""
    keys = ['t_move', 't_switch', 't_detect', 't_clear']
    lab = ['移动', '频道切换', '检测', '精确定位清除']
    cols = ['#63b3ed', '#f6ad55', '#68d391', '#f687b3']
    means = [np.mean([t[k] for t in trials]) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(1)
    for v, l, c in zip(means, lab, cols):
        ax.bar([0], [v], width=0.32, bottom=bottom, color=c, edgecolor='k', lw=0.9, label=l)
        bottom += np.array([v])
    ax.set_ylabel('平均时间 (s)')
    ax.set_xticks([])
    ax.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    ax.legend(fontsize=9, loc='upper right')
    despine(ax)
    save_fig(fig, '图3-3_定位清除时间构成')

def fig4_stats(trials):
    """清除个数与平均时间分布"""
    n_true = [t['n_true'] for t in trials]
    n_clear = [t['n_cleared'] for t in trials]
    avg_t = [t['time'] / max(t['n_cleared'], 1) for t in trials]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax1.hist(avg_t, bins=18, color='#68d391', edgecolor='k', lw=0.8, alpha=0.85)
    ax1.set_xlabel('平均定位清除时间 (s/个)'); ax1.set_ylabel('频数')
    ax1.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    ax2.bar(['真实数', '清除数'], [np.mean(n_true), np.mean(n_clear)], width=0.45,
            color=['#63b3ed', '#68d391'], edgecolor='k', lw=0.9)
    ax2.set_ylabel('平均个数')
    ax2.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    for a, lab in ((ax1, '(a) 平均定位清除时间分布'), (ax2, '(b) 清除个数')):
        despine(a); a.tick_params(direction='in')
        a.text(0.5, -0.2, lab, transform=a.transAxes, ha='center', fontsize=11, fontweight='bold')
    save_fig(fig, '图3-4_演练统计分布')

def grid_anchors(spacing=900.0):
    g = np.arange(-1350, 1351, spacing)
    A = [np.array([x, y]) for x in g for y in g]
    return A

def sparse_anchors():
    return [np.array([0.0, 0.0]), np.array([1150.0, 0.0]), np.array([-1150.0, 0.0]),
            np.array([0.0, 1150.0]), np.array([0.0, -1150.0])]

def fig5_compare():
    """策略对比：环形7锚点 vs 网格16锚点 vs 稀疏5锚点"""
    strategies = [('环形7锚点', anchors()), ('网格16锚点', grid_anchors()), ('稀疏5锚点', sparse_anchors())]
    res = {}
    for name, A in strategies:
        clears, times, trues = [], [], []
        for i in range(8):
            seed = 500 + i
            rng = np.random.default_rng(seed)
            sources = gen_case(rng)
            env = Simulator(sources, seed=seed + 1000)
            n_cleared, t, _ = run_strategy(env, seed + 2000, A=A)
            clears.append(n_cleared); times.append(t); trues.append(len(sources))
        res[name] = dict(clear_ratio=np.mean(clears)/np.mean(trues), avg_time=np.mean(times))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    names = [s[0] for s in strategies]
    cols = ['#63b3ed', '#68d391', '#f6ad55']
    ratio = [res[n]['clear_ratio'] * 100 for n in names]
    times = [res[n]['avg_time'] for n in names]
    ax1.bar(names, ratio, width=0.5, color=cols, edgecolor='k', lw=0.9)
    for i, v in enumerate(ratio):
        ax1.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=10)
    ax1.set_ylabel('清除比例 (%)')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    ax2.bar(names, times, width=0.5, color=cols, edgecolor='k', lw=0.9)
    for i, v in enumerate(times):
        ax2.text(i, v + 40, f'{v:.0f}', ha='center', fontsize=10)
    ax2.set_ylabel('平均总时间 (s)')
    ax2.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    for a, lab in ((ax1, '(a) 清除比例'), (ax2, '(b) 平均总时间')):
        despine(a); a.tick_params(direction='in')
        a.text(0.5, -0.2, lab, transform=a.transAxes, ha='center', fontsize=11, fontweight='bold')
    save_fig(fig, '图3-5_策略对比')

def fig6_timeline(sources, env):
    """清除时间轴（甘特）"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # 记录每个频道首次检测时刻与清除时刻
    first = {}; cleared_t = {}
    for act in env.log:
        action, pos, ch, res, t = act
        if action == 'measure' and ch not in first and res != 'no_signal':
            first[ch] = t
        if action == 'clear' and res == 'success' and ch not in cleared_t:
            cleared_t[ch] = t
    chs = sorted(set(list(first.keys()) + list(cleared_t.keys())))
    for i, ch in enumerate(chs):
        if ch in first:
            ax.barh(i, first[ch], left=0, height=0.5, color='#f6ad55', edgecolor='k', lw=0.6, alpha=0.85)
        if ch in cleared_t:
            ax.barh(i, cleared_t[ch] - first.get(ch, 0), left=first.get(ch, 0), height=0.5,
                    color='#68d391', edgecolor='k', lw=0.6, alpha=0.9)
    ax.set_yticks(range(len(chs)))
    ax.set_yticklabels([f'频道{ch}' for ch in chs], fontsize=8)
    ax.set_xlabel('虚拟时间 (s)')
    ax.grid(axis='x', lw=0.5, alpha=0.3, color='#94a3b8')
    despine(ax)
    save_fig(fig, '图3-6_各干扰源检测与清除时间轴')

# ---------------- 主流程 ----------------
if __name__ == '__main__':
    print('=' * 64)
    print('问题三：全向干扰源自动定位与清除')
    A = anchors()
    rng = np.random.default_rng(0)
    cov = coverage_check(rng, A)
    print(f'锚点数 {len(A)}，覆盖保证(源到最近锚点≤1000m)比例 = {cov*100:.2f}%')
    # 演练测试
    trials = [one_trial(i) for i in range(20)]
    clear_all = all(t['n_cleared'] == t['n_true'] for t in trials)
    print(f'演练 20 次：全部清除？ {clear_all}')
    print(f'  平均干扰源数 = {np.mean([t["n_true"] for t in trials]):.2f}')
    print(f'  平均清除数   = {np.mean([t["n_cleared"] for t in trials]):.2f}')
    avg_t = np.mean([t['time']/max(t['n_cleared'],1) for t in trials])
    print(f'  平均定位清除时间 = {avg_t:.2f} s/个')
    print(f'  平均总时间 = {np.mean([t["time"] for t in trials]):.1f} s')
    # 正式测试 3 次
    print('-' * 64)
    print('正式测试（自模拟数据）:')
    formal = []
    for k in range(3):
        seed = 9000 + k
        rng_f = np.random.default_rng(seed)
        sources = gen_case(rng_f)
        env = Simulator(sources, seed=seed + 1000)
        n_cleared, t, detail = run_strategy(env, seed + 2000)
        formal.append((seed, sources, env, n_cleared, t))
        print(f'  测试{k+1}: 案例编码 B3-S{seed}, 清除 {n_cleared}/{len(sources)}, '
              f'平均定位清除时间 {t/max(n_cleared,1):.2f} s/个, 总时间 {t:.1f} s')

    # 图
    fig1_trajectory(None, 9000)
    fig2_loc_error()
    fig3_time_composition(trials)
    fig4_stats(trials)
    fig5_compare()
    seed, sources, env, n_cleared, t = formal[0]
    fig6_timeline(sources, env)

    # CSV：演练统计
    df_tr = pd.DataFrame([{'案例': i+1, '干扰源总数': t['n_true'], '清除个数': t['n_cleared'],
                           '总时间(s)': round(t['time'], 1), '平均定位清除时间(s)': round(t['time']/max(t['n_cleared'],1), 1),
                           '移动(s)': round(t['t_move'],1), '切换(s)': round(t['t_switch'],1),
                           '检测(s)': round(t['t_detect'],1), '清除(s)': round(t['t_clear'],1)} for i, t in enumerate(trials)])
    df_tr.to_csv(os.path.join(OUT, '问题三_演练统计.csv'), index=False, encoding='utf-8-sig')
    # 正式测试结果（表1）
    df_fm = pd.DataFrame([{'测试案例编码': f'B3-S{s}', '清除干扰源个数': n, '平均定位清除时间(s)': round(tt/max(n,1),1),
                           '程序运行时间(s)': round(tt,1)} for (s, _, _, n, tt) in formal])
    df_fm.to_csv(os.path.join(OUT, '问题三_正式测试结果.csv'), index=False, encoding='utf-8-sig')
    # 行为日志（正式测试1）
    seed, sources, env, n_cleared, t = formal[0]
    log_df = pd.DataFrame([{'步骤': i+1, '动作': a, 'x(m)': round(p[0],2), 'y(m)': round(p[1],2),
                            '频道': ch, '结果': (r[0] if isinstance(r, tuple) else r), '示向度(deg)': (r[1] if isinstance(r, tuple) else ''),
                            '累计虚拟时间(s)': tt} for i, (a, p, ch, r, tt) in enumerate(env.log)])
    log_df.to_csv(os.path.join(OUT, '正式测试1_日志.csv'), index=False, encoding='utf-8-sig')
    print('已输出：问题三_演练统计.csv / 问题三_正式测试结果.csv / 正式测试1_日志.csv')
    print('全部图片与结果输出完毕。')
