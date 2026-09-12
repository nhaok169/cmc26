# -*- coding: utf-8 -*-
"""
问题四：含定向干扰源的机器狗自动定位与清除策略
定向源仅在"定向方向±90°"半平面内辐射。采用分层环形锚点（内环12@900 + 中环16@1750 + 外环24@2600）
保证任意定向源（任意位置、任意朝向）至少一个锚点落入其半平面且距离≤1000m（蒙特卡洛检测率≈100%）。
策略：分层覆盖扫描 → 示向度交会定位/沿示向线二分 → TSP清除 + 就近寻的 → 复扫验证。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import *
import itertools, math

FIG = FIG_DIR
OUT = OUT_DIR
ARENA_R = 1800.0
V = 5.0
T_DETECT = 5.0
T_SWITCH = 1.0
T_CLEAR_OK = 5.0
T_CLEAR_FAIL = 3.0
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

# ---------------- 模拟器（含定向源） ----------------
class Simulator:
    def __init__(self, sources, seed=0):
        self.sources = sources  # ch -> dict(pos, R, type, dir)
        self.pos = np.array([0.0, 0.0])
        self.channel = 1
        self.time = 0.0
        self.cleared = set()
        self.rng = np.random.default_rng(seed)
        self.t_move = 0.0; self.t_switch = 0.0; self.t_detect = 0.0; self.t_clear = 0.0
        self.n_detect = 0; self.n_clear = 0
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
            self.time += T_SWITCH; self.t_switch += T_SWITCH
            self.channel = ch
        self.time += T_DETECT; self.t_detect += T_DETECT; self.n_detect += 1
        src = self.sources.get(ch)
        if src is None or ch in self.cleared:
            res = 'no_signal'
        else:
            G = src['pos']; R = src['R']
            d = np.linalg.norm(p - G)
            if d <= R:
                if src['type'] == 'directional' and (p - G) @ src['dir'] < 0:
                    res = 'no_signal'   # 不在覆盖半平面
                elif d <= NEAR_R:
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
    n_dir = int(rng.integers(1, n_src))          # 既有全向又有定向：1~n-1 个定向
    dirs = set(rng.choice(chs, size=n_dir, replace=False).tolist())
    sources = {}
    for ch in chs:
        ang = rng.uniform(0, 2 * np.pi)
        rr = ARENA_R * np.sqrt(rng.uniform(0, 1))
        pos = np.array([rr * np.cos(ang), rr * np.sin(ang)])
        R = rng.uniform(1000.0, 1500.0)
        ch = int(ch)
        if ch in dirs:
            d = unit(rng.uniform(0, 360))
            sources[ch] = {'pos': pos, 'R': R, 'type': 'directional', 'dir': d}
        else:
            sources[ch] = {'pos': pos, 'R': R, 'type': 'omni', 'dir': None}
    return sources

# ---------------- 锚点 ----------------
def ring(rho, n):
    return [rho * np.array([np.cos(np.deg2rad(360.0 * k / n)), np.sin(np.deg2rad(360.0 * k / n))]) for k in range(n)]

def anchors():
    A = [np.array([0.0, 0.0])] + ring(900.0, 12) + ring(1750.0, 12) + ring(2600.0, 24)
    return A

def detectable(sources, A):
    """检查每个源是否至少一个锚点落入其半平面且≤1000m"""
    miss = 0
    for ch, s in sources.items():
        ok = False
        for P in A:
            if np.linalg.norm(P - s['pos']) <= 1000.0 and \
               (s['type'] != 'directional' or (P - s['pos']) @ s['dir'] >= 0):
                ok = True; break
        if not ok:
            miss += 1
    return miss

# ---------------- 定位/清除辅助 ----------------
def localize(bearings):
    A = np.zeros((2, 2)); b = np.zeros(2)
    for P, th in bearings:
        n = left_normal(th)
        A += np.outer(n, n)
        b += (n @ P) * n
    return np.linalg.solve(A, b)

def tsp_order(pts, start):
    order = []; cur = np.asarray(start, float)
    rem = set(range(len(pts)))
    while rem:
        j = min(rem, key=lambda k: np.linalg.norm(pts[k] - cur))
        order.append(j); cur = pts[j]; rem.remove(j)
    return order

def grid_clear(env, ch, pos, rings=((0.0, 1), (20.0, 6), (40.0, 10), (60.0, 14))):
    for r, n in rings:
        for k in range(n):
            th = 2*np.pi*k/max(n, 1)
            p = pos + r*np.array([np.cos(th), np.sin(th)])
            if env.clear(p[0], p[1], ch) == 'success':
                return True
    return False

def max_crossing(bearings):
    """示向线两两交会角的最大值（交会角=两条线的锐角）"""
    ths = [t for _, t in bearings]
    best = 0.0
    for i in range(len(ths)):
        for j in range(i + 1, len(ths)):
            d = abs(ths[i] - ths[j]) % 180.0
            best = max(best, min(d, 180.0 - d))
    return best

def home_bisect(env, ch, P, th):
    """沿示向线二分（定向源从正面接近，越过源后 no_signal）。源距≤1500（有效半径上限）"""
    u = unit(th)
    lo, hi = 5.0, 1500.0
    for _ in range(15):
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
            lo = t
        else:
            hi = t
    pos = P + ((lo + hi) / 2) * u
    return grid_clear(env, ch, pos)

def clear_target(env, ch, est):
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
    return grid_clear(env, ch, pos)

# ---------------- 策略 ----------------
def run_strategy(env, seed):
    A = anchors()
    active = set(range(1, 21))
    bearings = {ch: [] for ch in range(1, 21)}
    estimates = {}
    # Phase 1 分层覆盖扫描
    for a in A:
        for ch in sorted(active):
            r = env.measure(a[0], a[1], ch)
            if r == 'near':
                if env.clear(a[0], a[1], ch) == 'success':
                    active.discard(ch)
            elif r != 'no_signal' and r[0] == 'direction':
                bearings[ch].append((a.copy(), r[1]))
                if len(bearings[ch]) >= 2 and max_crossing(bearings[ch]) >= 25.0:
                    estimates[ch] = localize(bearings[ch])
                    active.discard(ch)
        if not active:
            break
    # Phase 1.5 单示向度源二分；≥2 条示向度源交会定位
    # （分层锚点对任意定向源覆盖检测率≈100%，0 示向度频道即无源）
    for ch in list(active):
        if len(bearings[ch]) == 1:
            P, th = bearings[ch][0]
            if home_bisect(env, ch, P, th):
                active.discard(ch)
        elif len(bearings[ch]) >= 2:
            estimates[ch] = localize(bearings[ch])
            active.discard(ch)
    # Phase 2 清除路径
    if estimates:
        keys = list(estimates.keys())
        pts = [estimates[k] for k in keys]
        for j in tsp_order(pts, env.pos):
            ch = keys[j]
            ok = clear_target(env, ch, estimates[ch])
            if not ok and len(bearings[ch]) >= 1:
                # 交会估计失效（定向源从背面接近等）→ 用第一条示向度（来自正面锚点）二分
                P, th = bearings[ch][0]
                ok = home_bisect(env, ch, P, th)
            if ok and ch in active:
                active.discard(ch)
    # Phase 3 验证：未确认频道中心复扫
    for ch in sorted(range(1, 21)):
        if ch in env.cleared:
            continue
        r = env.measure(0.0, 0.0, ch)
        if r == 'near':
            env.clear(0.0, 0.0, ch)
    return len(env.cleared), env.time, dict(t_move=env.t_move, t_switch=env.t_switch,
                                            t_detect=env.t_detect, t_clear=env.t_clear)

def one_trial(seed):
    rng = np.random.default_rng(seed)
    sources = gen_case(rng)
    n_dir = sum(1 for s in sources.values() if s['type'] == 'directional')
    env = Simulator(sources, seed=seed + 1000)
    n_cleared, t, detail = run_strategy(env, seed + 2000)
    return dict(n_true=len(sources), n_clear=n_cleared, n_dir=n_dir, time=t, **detail)

# ---------------- 图 ----------------
def fig1_directional():
    """定向源半平面覆盖示意"""
    G = np.array([1000.0, 600.0])
    d = unit(40.0)  # 定向方向 40°
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(Circle((0, 0), ARENA_R, fill=False, ec='#94a3b8', lw=1.0, ls='--'))
    R = 1300.0
    ax.add_patch(Circle(G, R, fill=False, ec='#2b6cb0', lw=1.1, ls='-'))
    # 半平面（±90° 覆盖）
    L = 2000.0
    for dd in (-90.0, 90.0):
        v = unit(40.0 + dd)
        ax.plot([G[0], G[0] + L*v[0]], [G[1], G[1] + L*v[1]], color='#d97706', lw=1.2, ls='--')
    # 半平面内填充
    t = np.linspace(-90, 90, 120)
    pts = G + L*np.stack([unit(40.0+dd) for dd in t])
    poly = np.vstack([[G], pts])
    ax.fill(poly[:, 0], poly[:, 1], color='#f6ad55', alpha=0.25, lw=0)
    # 定向方向
    ax.annotate('', xy=G + 900*unit(40), xytext=G, arrowprops=dict(arrowstyle='-|>', color='#e53e3e', lw=2.2))
    ax.annotate('定向方向 d', (G[0]+900*unit(40)[0]-20, G[1]+900*unit(40)[1]+40), color='#9b2c2c', fontsize=11)
    # 可检测点 vs 不可检测点
    det = [G + 700*unit(20), G + 800*unit(70), G + 1000*unit(110)]
    nodet = [G + 700*unit(200), G + 900*unit(250), G + 800*unit(330)]
    for p in det:
        ax.plot([p[0]], [p[1]], 'o', ms=9, color='#2f855a', mec='k', mew=0.9)
    for p in nodet:
        ax.plot([p[0]], [p[1]], 'x', ms=11, color='#e53e3e', mew=2.2)
    ax.plot([G[0]], [G[1]], '*', ms=20, color='#2b6cb0', mec='k', mew=1.0, zorder=5)
    ax.annotate('G', (G[0]+30, G[1]+30), fontsize=12, color='#1e3a8a')
    ax.annotate('可检测(绿)', (G[0]-150, G[1]+850), fontsize=9, color='#276749')
    ax.annotate('不可检测(红×)', (G[0]-150, G[1]+780), fontsize=9, color='#9b2c2c')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(-1900, 1900); ax.set_ylim(-1900, 1900)
    despine(ax)
    save_fig(fig, '图4-1_定向源半平面覆盖示意')

def fig2_anchors():
    """分层环形锚点布局"""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(Circle((0, 0), ARENA_R, fill=False, ec='#94a3b8', lw=1.0, ls='--'))
    for rho, n, c, lab in ((900, 12, '#f6ad55', '内环 12'), (1750, 12, '#68d391', '中环 12'), (2600, 24, '#63b3ed', '外环 24')):
        R = ring(rho, n)
        ax.plot([p[0] for p in R], [p[1] for p in R], 'o', ms=6, color=c, mec='k', mew=0.7, zorder=4, label=f'{lab} (ρ={rho}m)')
        ax.add_patch(Circle((0, 0), rho, fill=False, ec=c, lw=0.7, ls=':', alpha=0.7))
    ax.plot([0], [0], 'o', ms=8, color='#e53e3e', mec='k', mew=1.0, zorder=5, label='中心')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(-2900, 2900); ax.set_ylim(-2900, 2900)
    ax.legend(fontsize=9, loc='upper right')
    despine(ax)
    save_fig(fig, '图4-2_分层环形锚点布局')

def fig3_trajectory(seed=9000):
    rng = np.random.default_rng(seed)
    sources = gen_case(rng)
    env = Simulator(sources, seed=seed + 1000)
    n_cleared, t, detail = run_strategy(env, seed + 2000)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(Circle((0, 0), ARENA_R, fill=False, ec='#94a3b8', lw=1.0, ls='--'))
    for ch, s in sources.items():
        if s['type'] == 'directional':
            ax.plot([s['pos'][0]], [s['pos'][1]], 'D', ms=9, color='#e53e3e', mec='k', mew=0.8, zorder=5)
            ax.annotate('', xy=s['pos'] + 220*s['dir'], xytext=s['pos'],
                        arrowprops=dict(arrowstyle='-|>', color='#e53e3e', lw=1.3))
        else:
            ax.plot([s['pos'][0]], [s['pos'][1]], 'o', ms=9, color='#2f855a', mec='k', mew=0.8, zorder=5)
    xs = [0.0]; ys = [0.0]
    for act in env.log:
        xs.append(act[1][0]); ys.append(act[1][1])
    ax.plot(xs, ys, color='#2b6cb0', lw=1.0, alpha=0.85, zorder=3)
    ax.plot([0], [0], '^', ms=10, color='#2b6cb0', mec='k', mew=1.0, zorder=6)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(-2900, 2900); ax.set_ylim(-2900, 2900)
    despine(ax)
    save_fig(fig, '图4-3_干扰源分布与机器狗轨迹')

def fig4_detection_rate():
    """分层锚点检测率"""
    rng = np.random.default_rng(5)
    sets = [('仅内环12', ring(900, 12)),
            ('内环12+中环12', ring(900, 12) + ring(1750, 12)),
            ('内环12+中环12+外环24', ring(900, 12) + ring(1750, 12) + ring(2600, 24))]
    rates = []
    for name, A in sets:
        miss = 0; n = 15000
        for _ in range(n):
            ang = rng.uniform(0, 2*np.pi); rr = ARENA_R*np.sqrt(rng.uniform(0, 1))
            G = np.array([rr*np.cos(ang), rr*np.sin(ang)]); d = unit(rng.uniform(0, 360))
            ok = False
            for P in A:
                if np.linalg.norm(P - G) <= 1000 and (P - G) @ d >= 0:
                    ok = True; break
            if not ok:
                miss += 1
        rates.append((1 - miss/n) * 100)
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [s[0] for s in sets]
    ax.bar(names, rates, width=0.5, color=['#f6ad55', '#68d391', '#63b3ed'], edgecolor='k', lw=0.9)
    for i, v in enumerate(rates):
        ax.text(i, v + 1.5, f'{v:.3f}%', ha='center', fontsize=10)
    ax.set_ylabel('定向源检测率 (%)')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    despine(ax)
    save_fig(fig, '图4-4_分层锚点检测率')

def fig5_time_composition(trials):
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
    save_fig(fig, '图4-5_定位清除时间构成')

def fig6_stats(trials):
    avg_t = [t['time']/max(t['n_clear'], 1) for t in trials]
    n_dir = [t['n_dir'] for t in trials]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax1.hist(avg_t, bins=16, color='#68d391', edgecolor='k', lw=0.8, alpha=0.85)
    ax1.set_xlabel('平均定位清除时间 (s/个)'); ax1.set_ylabel('频数')
    ax1.grid(axis='y', lw=0.5, alpha=0.3, color='#94a3b8')
    ax2.scatter(n_dir, avg_t, s=60, color='#63b3ed', edgecolor='k', lw=0.7, zorder=3)
    ax2.set_xlabel('定向干扰源个数'); ax2.set_ylabel('平均定位清除时间 (s/个)')
    ax2.grid(lw=0.5, alpha=0.3, color='#94a3b8')
    for a, lab in ((ax1, '(a) 平均定位清除时间分布'), (ax2, '(b) 时间与定向源个数关系')):
        despine(a); a.tick_params(direction='in')
        a.text(0.5, -0.2, lab, transform=a.transAxes, ha='center', fontsize=11, fontweight='bold')
    save_fig(fig, '图4-6_演练统计分布')

# ---------------- 主流程 ----------------
if __name__ == '__main__':
    print('=' * 64)
    print('问题四：含定向干扰源的自动定位与清除')
    A = anchors()
    print(f'锚点数 = {len(A)}（中心+内环12@900+中环12@1750+外环24@2600）')
    # 覆盖检测率
    rng = np.random.default_rng(5)
    miss = 0; ntest = 20000
    for _ in range(ntest):
        ang = rng.uniform(0, 2*np.pi); rr = ARENA_R*np.sqrt(rng.uniform(0, 1))
        G = np.array([rr*np.cos(ang), rr*np.sin(ang)]); d = unit(rng.uniform(0, 360))
        ok = False
        for P in A:
            if np.linalg.norm(P - G) <= 1000 and (P - G) @ d >= 0:
                ok = True; break
        if not ok:
            miss += 1
    print(f'定向源覆盖检测率（蒙特卡洛 {ntest} 次）= {(1-miss/ntest)*100:.4f}%')
    # 演练测试
    trials = [one_trial(i) for i in range(20)]
    clear_all = all(t['n_clear'] == t['n_true'] for t in trials)
    print(f'演练 20 次：全部清除？ {clear_all}')
    print(f'  平均干扰源数 = {np.mean([t["n_true"] for t in trials]):.2f}（其中定向 {np.mean([t["n_dir"] for t in trials]):.2f}）')
    print(f'  平均清除数   = {np.mean([t["n_clear"] for t in trials]):.2f}')
    print(f'  平均定位清除时间 = {np.mean([t["time"]/max(t["n_clear"],1) for t in trials]):.2f} s/个')
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
        print(f'  测试{k+1}: 案例编码 B4-S{seed}, 清除 {n_cleared}/{len(sources)}, '
              f'平均定位清除时间 {t/max(n_cleared,1):.2f} s/个, 总时间 {t:.1f} s')

    # 图
    fig1_directional()
    fig2_anchors()
    fig3_trajectory(9000)
    fig4_detection_rate()
    fig5_time_composition(trials)
    fig6_stats(trials)

    # CSV
    df_tr = pd.DataFrame([{'案例': i+1, '干扰源总数': t['n_true'], '定向源数': t['n_dir'], '清除个数': t['n_clear'],
                           '总时间(s)': round(t['time'], 1), '平均定位清除时间(s)': round(t['time']/max(t['n_clear'],1), 1),
                           '移动(s)': round(t['t_move'],1), '切换(s)': round(t['t_switch'],1),
                           '检测(s)': round(t['t_detect'],1), '清除(s)': round(t['t_clear'],1)} for i, t in enumerate(trials)])
    df_tr.to_csv(os.path.join(OUT, '问题四_演练统计.csv'), index=False, encoding='utf-8-sig')
    df_fm = pd.DataFrame([{'测试案例编码': f'B4-S{s}', '清除干扰源个数': n, '平均定位清除时间(s)': round(tt/max(n,1),1),
                           '程序运行时间(s)': round(tt,1)} for (s, _, _, n, tt) in formal])
    df_fm.to_csv(os.path.join(OUT, '问题四_正式测试结果.csv'), index=False, encoding='utf-8-sig')
    seed, sources, env, n_cleared, t = formal[0]
    log_df = pd.DataFrame([{'步骤': i+1, '动作': a, 'x(m)': round(p[0],2), 'y(m)': round(p[1],2),
                            '频道': ch, '结果': (r[0] if isinstance(r, tuple) else r), '示向度(deg)': (r[1] if isinstance(r, tuple) else ''),
                            '累计虚拟时间(s)': tt} for i, (a, p, ch, r, tt) in enumerate(env.log)])
    log_df.to_csv(os.path.join(OUT, '正式测试1_日志.csv'), index=False, encoding='utf-8-sig')
    print('已输出：问题四_演练统计.csv / 问题四_正式测试结果.csv / 正式测试1_日志.csv')
    print('全部图片与结果输出完毕。')
