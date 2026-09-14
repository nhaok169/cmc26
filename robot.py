# -*- coding: utf-8 -*-
"""
2026 高教社杯 B 题 机器狗程序（问题3 / 问题4 简版策略）

用法:
    python robot.py --team <参赛队号> --problem 3 [--port 2026] [--wait 600]
    python robot.py --team <参赛队号> --problem 4

策略概述:
  1. 等待测试窗口开启(轮询 /enter, 复用同一 request_id, 幂等安全)
  2. 扫描阶段: 依次走到 13/17 个扫描点, 在每个点对未清除频道逐个测向
     扫描点 = 中心 + 6@r1000 + 6@r1600(偏移30°), 已数值验证:
     区域内任一点到最近扫描点距离 <= 801m < 1000m(接收半径下限)
  3. 定位阶段: 对每个听到信号的频道, 补充 2~3 次不同位置的测向,
     所得方位线做最小二乘交会 -> 估计干扰源位置
  4. 清除阶段: 走到估计点 /clear; 若失败按 25m 螺旋扩大搜索清除
  5. 问题4: 定向干扰源存在盲区, 用"换几何形状的第二轮扫描"兜底,
     连续 2 轮无新发现则结束
  6. 输出统计: 清除数 / 虚拟总时间 / 平均定位清除时间, 并写日志
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import requests

# ---------------- 常量 ----------------
SPEED = 5.0            # m/s
SWITCH_COST = 1.0      # s
MEASURE_COST = 5.0     # s
CLEAR_COST = 3.0       # s (未发现) / 成功为5
NEAR_R = 5.0           # 近距离阈值
CLEAR_R = 20.0         # 清除半径
ARENA_R = 1800.0
RECV_R_MIN = 1000.0    # 有效接收半径下限(规划用)
ERR_DEG = 1.0          # 测向误差 ±1°
CHANNELS = list(range(1, 21))
REAL_TIME_LIMIT = 18 * 60  # 程序运行自保上限(秒), 留裕量给 /exit


def pol(r, a_deg):
    return (r * math.cos(math.radians(a_deg)), r * math.sin(math.radians(a_deg)))


def scan_points(rot=0.0):
    """区域内 13 个扫描点; rot 为整体旋转角(度)"""
    pts = [(0.0, 0.0)]
    for i in range(6):
        pts.append(pol(1000, i * 60 + rot))
    for i in range(6):
        pts.append(pol(1600, i * 60 + 30 + rot))
    return pts


def outer_points(rot=15.0):
    """区域外圈 12 个扫描点(r=2000, 允许在区域外)。
    作用: 兜住"贴边且朝向朝外"的定向干扰源 —— 它们的覆盖半平面
    几乎全部在目标区域外, 区域内的扫描点全部处于盲区。
    数值验证: 内圈 + 外圈对随机定向源 20 万样本 0 漏检。"""
    return [pol(2000, i * 30 + rot) for i in range(12)]


def greedy_path(start, pts):
    """简单最近邻排序, 减少移动耗时"""
    remain = list(pts)
    cur = start
    order = []
    while remain:
        nxt = min(remain, key=lambda p: math.hypot(p[0] - cur[0], p[1] - cur[1]))
        order.append(nxt)
        remain.remove(nxt)
        cur = nxt
    return order


class TestEnded(Exception):
    """模拟器返回 accepted=false(测试已结束/被拒), 需要优雅收尾"""
    pass


class Sim:
    """模拟器 HTTP 客户端: 串行动作, 幂等重试"""

    def __init__(self, base, team, log):
        self.base = base
        self.team = team
        self.log = log
        self.n = 0
        self.cur_channel = 1      # /enter 后测向机在频道1
        self.cur_pos = (0.0, 0.0)
        self.vt = 0.0             # 最近一次 accepted 的虚拟时刻
        self.t0 = None            # /enter 成功的现实时刻
        self.cleared = set()
        self.measure_cnt = 0
        self.switch_cnt = 0
        self.clear_fail = 0

    def _rid(self, tag):
        self.n += 1
        return f"{tag}-{self.n}"

    def _post(self, path, body, rid, timeout=15):
        """幂等 POST: 网络异常/5xx/429 复用同一 request_id 重试"""
        last = None
        for attempt in range(10):
            try:
                r = requests.post(self.base + path, json=body, timeout=timeout,
                                  headers={"Content-Type": "application/json; charset=utf-8"})
                if r.status_code in (429, 500):
                    # 瞬时服务端错误/限流: 幂等重试同一动作
                    last = f"HTTP {r.status_code}"
                    time.sleep(1.5)
                    continue
                try:
                    return r.status_code, r.json()
                except ValueError:
                    last = f"HTTP {r.status_code}, non-JSON body"
            except requests.RequestException as e:
                last = repr(e)
            time.sleep(0.5)
        raise RuntimeError(f"请求 {path} {rid} 多次失败: {last}")

    def act(self, path, body, tag, timeout=15):
        rid = self._rid(tag)
        body = dict(body, arena_id="default", robot_id=self.team, request_id=rid)
        code, resp = self._post(path, body, rid, timeout)
        self.log.write(json.dumps({"t": datetime.datetime.now().isoformat(timespec="seconds"),
                                   "rid": rid, "path": path, "req": body,
                                   "http": code, "resp": resp}, ensure_ascii=False) + "\n")
        self.log.flush()
        if code == 200 and resp.get("accepted"):
            if path in ("/measure", "/clear"):
                self.vt = resp["virtual_time_s"]
            return resp
        # accepted=false: 多为测试已结束/窗口超时; 优雅收尾而不是崩溃
        raise TestEnded(f"动作被拒绝 {path} {rid}: HTTP {code} {resp}")

    def try_enter(self, request_id):
        """试一次 /enter; 成功返回 resp, 否则 (None, 原因)。"""
        body = {"arena_id": "default", "robot_id": self.team, "request_id": request_id}
        try:
            r = requests.post(self.base + "/enter", json=body, timeout=5,
                              headers={"Content-Type": "application/json; charset=utf-8"})
            code, resp = r.status_code, None
            try:
                resp = r.json()
            except ValueError:
                pass
        except requests.RequestException as e:
            return None, repr(e)
        if code == 200 and resp and resp.get("accepted"):
            self.t0 = time.time()
            self.vt = resp["virtual_time_s"]
            self.log.write(json.dumps({"t": datetime.datetime.now().isoformat(),
                                       "rid": request_id, "path": "/enter",
                                       "http": code, "resp": resp}) + "\n")
            self.log.flush()
            print(f"[ENTER] ok, 剩余现实时间 {resp.get('remaining_real_duration_s')}s")
            return resp, None
        return None, f"HTTP {code} {resp}"

    def enter(self, wait_s=600, request_id=None):
        """等待测试窗口开启并进入; 同一次等待复用同一 request_id。
        wait_s<=0 表示一直等到成功。"""
        rid = request_id or "enter-1"
        t_end = (time.time() + wait_s) if wait_s and wait_s > 0 else None
        last = None
        while True:
            resp, last = self.try_enter(rid)
            if resp is not None:
                return resp
            if t_end is not None and time.time() > t_end:
                raise RuntimeError(f"等待测试窗口超时({wait_s}s): {last}")
            time.sleep(1.0)

    def measure(self, pos, ch):
        x, y = round(float(pos[0]), 3), round(float(pos[1]), 3)
        resp = self.act("/measure", {"position": {"x": x, "y": y}, "channel": ch}, "m")
        if ch != self.cur_channel:
            self.switch_cnt += 1
        self.cur_channel = ch
        self.cur_pos = (x, y)
        self.measure_cnt += 1
        return resp

    def clear(self, pos, ch):
        x, y = round(float(pos[0]), 3), round(float(pos[1]), 3)
        resp = self.act("/clear", {"position": {"x": x, "y": y}, "channel": ch}, "c")
        self.cur_pos = (x, y)
        ok = resp.get("clear_result") == "success"
        if ok:
            self.cleared.add(ch)
        else:
            self.clear_fail += 1
        return ok

    def exit(self):
        return self.act("/exit", {}, "x")

    def elapsed_real(self):
        return time.time() - self.t0 if self.t0 else 0.0


def watch_loop(base, team, problem, make_policy, label):
    """持续监听官方窗口: 进局 → 跑策略 → 退出 → 再等下一局。Ctrl+C 停止。"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"watch: robot_id={team}, 问题{problem}, 接口 {base}")
    print(f"策略: {label}")
    print("持续等待测试窗口, Ctrl+C 停止")
    n_fail = 0
    game = 0
    while True:
        game += 1
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"robot_p{problem}_{stamp}.jsonl")
        log = open(log_path, "w", encoding="utf-8")
        sim = Sim(base, team, log)
        rid = f"enter-{stamp}-{game}"
        while True:
            resp, err = sim.try_enter(rid)
            if resp is not None:
                n_fail = 0
                remain = resp.get("remaining_real_duration_s")
                print(f"\n==== 第{game}局 remaining={remain}s "
                      f"{time.strftime('%H:%M:%S')} ====")
                print(f"日志: {log_path}")
                break
            n_fail += 1
            if n_fail <= 2 or n_fail % 8 == 0:
                print(f"enter wait ({n_fail}): {err}")
            time.sleep(3.5)
        try:
            n, vt = make_policy(sim).run()
            avg = (vt / n) if n else float("inf")
            print(f"[DONE] cleared={n} vt={vt:.1f} avg={avg:.1f}s")
        except TestEnded as e:
            print(f"[END] {e}")
        except Exception as e:
            print(f"[ERR] {type(e).__name__}: {e}")
        try:
            sim.exit()
            print("[EXIT] 已退出")
        except Exception as e:
            print(f"[EXIT] {e}")
        log.close()
        print("等待下一局...")
        time.sleep(2.0)


# ---------------- 定位数学 ----------------
def bearing_line(p, deg):
    """返回 (A, u): 过点A、方向u 的方位线"""
    a = math.radians(deg)
    return p, (math.cos(a), math.sin(a))


def lsq_estimate(lines):
    """最小二乘交会: 每条线给 n·x = n·A, n 为法向"""
    Sxx = Sxy = Syy = sx = sy = 0.0
    for (A, u) in lines:
        n = (-u[1], u[0])
        b = n[0] * A[0] + n[1] * A[1]
        Sxx += n[0] * n[0]; Sxy += n[0] * n[1]; Syy += n[1] * n[1]
        sx += n[0] * b; sy += n[1] * b
    det = Sxx * Syy - Sxy * Sxy
    if abs(det) < 1e-9:
        return None
    return ((sx * Syy - sy * Sxy) / det, (Sxx * sy - Sxy * sx) / det)


def dist_to_line(p, line):
    (A, u) = line
    dx, dy = p[0] - A[0], p[1] - A[1]
    return abs(dx * (-u[1]) + dy * u[0])


# ---------------- 策略 ----------------
class Robot:
    def __init__(self, sim, problem):
        self.sim = sim
        self.problem = problem
        self.bearings = {}   # ch -> [(pos, deg)]

    # -- 单目标: 补测 -> 交会 -> 清除 --
    def hunt(self, ch):
        sim = self.sim
        lines = [bearing_line(p, d) for (p, d) in self.bearings[ch]]
        est = None
        # 补测至多 4 次: 优先垂直偏移(交角大), 无信号则沿方位线推进
        for k in range(4):
            if len(lines) >= 2:
                est = lsq_estimate(lines)
                res = max(dist_to_line(est, L) for L in lines)
                if len(lines) >= 3 and res < 15.0:
                    break
                # 已有估计: 到估计点前 300m、侧偏 150m 处补测(交角好且近)
                dx, dy = est[0] - sim.cur_pos[0], est[1] - sim.cur_pos[1]
                dd = math.hypot(dx, dy) or 1.0
                u = (dx / dd, dy / dd)
                pp = (-u[1], u[0])
                q = (est[0] - 300 * u[0] + 150 * pp[0], est[1] - 300 * u[1] + 150 * pp[1])
            else:
                # 只有一条方位线: 先两侧垂直偏移 600m, 无信号则沿线推进
                (A0, u0) = lines[0]
                perp = (-u0[1], u0[0])
                if k == 0:
                    q = (A0[0] + 600 * perp[0], A0[1] + 600 * perp[1])
                elif k == 1:
                    q = (A0[0] - 600 * perp[0], A0[1] - 600 * perp[1])
                else:
                    d_along = (800.0, 1300.0)[k - 2]
                    q = (A0[0] + d_along * u0[0], A0[1] + d_along * u0[1])
            r = sim.measure(q, ch)
            mr = r.get("measure_result")
            if mr == "near":
                # 距离<=5m, 直接原地清除(清除半径20m)
                if sim.clear(sim.cur_pos, ch):
                    print(f"  [ch{ch}] 近距离直接清除 @ {sim.cur_pos}")
                    return True
            elif mr == "direction":
                self.bearings[ch].append((sim.cur_pos, r["svd_deg"]))
                lines.append(bearing_line(sim.cur_pos, r["svd_deg"]))
            # no_signal: 换下一个候选点重试
        est = lsq_estimate(lines) if len(lines) >= 2 else None
        if est is None:
            # 退化: 只有一条方位线, 沿线走 800m 试探
            (A0, u0) = lines[0]
            est = (A0[0] + 800 * u0[0], A0[1] + 800 * u0[1])
        # 清除: 估计点 -> 螺旋兜底
        if sim.clear(est, ch):
            print(f"  [ch{ch}] 交会估计 {est[0]:.0f},{est[1]:.0f} 清除成功 (vt={sim.vt:.0f}s)")
            return True
        # 螺旋: 半径25,50,...,175, 每圈6点
        for rad in range(25, 200, 25):
            for a in range(0, 360, 60):
                q = (est[0] + rad * math.cos(math.radians(a)),
                     est[1] + rad * math.sin(math.radians(a)))
                if sim.clear(q, ch):
                    print(f"  [ch{ch}] 螺旋搜索 r={rad} 清除成功 (vt={sim.vt:.0f}s)")
                    return True
        print(f"  [ch{ch}] 清除失败!")
        return False

    # -- 一轮扫描: 输入扫描点列表, 返回本轮新听到(未清除)的频道 --
    def sweep(self, pts):
        sim = self.sim
        found = []
        pts = greedy_path(sim.cur_pos, pts)
        np_ = len(pts)
        for i, p in enumerate(pts):
            for ch in CHANNELS:
                if ch in sim.cleared:
                    continue
                # 现实时间自保
                if sim.elapsed_real() > REAL_TIME_LIMIT:
                    print(f"[WARN] 现实时间接近上限, 提前结束扫描")
                    return found, True
                r = sim.measure(p, ch)
                mr = r.get("measure_result")
                if mr == "direction":
                    self.bearings.setdefault(ch, []).append((sim.cur_pos, r["svd_deg"]))
                    if ch not in found:
                        found.append(ch)
                    print(f"  [扫描点{i+1}/{np_}] ch{ch} 方向 {r['svd_deg']:.2f}° @ {p[0]:.0f},{p[1]:.0f} (vt={sim.vt:.0f}s)")
                elif mr == "near":
                    if sim.clear(sim.cur_pos, ch):
                        print(f"  [扫描点{i+1}/{np_}] ch{ch} 近距离直接清除 (vt={sim.vt:.0f}s)")
        return found, False

    def run(self):
        sim = self.sim
        # 问题3: 全向源无盲区, 1 轮内圈扫描即全覆盖(几何保证)
        # 问题4: 内圈 -> 区域外圈(兜贴边朝外定向源) -> 旋转30°内圈(保险)
        if self.problem == 3:
            round_sets = [scan_points(0.0)]
        else:
            round_sets = [scan_points(0.0), outer_points(15.0), scan_points(30.0)]
        rounds_no_new = 0
        try:
            for ri, pts in enumerate(round_sets):
                print(f"\n===== 扫描轮 {ri+1}/{len(round_sets)} ({len(pts)}个点) =====")
                found, abort = self.sweep(pts)
                # 逐个清除本轮发现的目标
                pend = [c for c in found if c not in sim.cleared]
                while pend:
                    if sim.elapsed_real() > REAL_TIME_LIMIT:
                        break
                    ch = pend[0]
                    ok = self.hunt(ch)
                    pend = [c for c in pend[1:] if c not in sim.cleared] if not ok \
                        else [c for c in pend if c not in sim.cleared]
                if not found:
                    rounds_no_new += 1
                else:
                    rounds_no_new = 0
                if abort or rounds_no_new >= 2 or sim.elapsed_real() > REAL_TIME_LIMIT:
                    break
        except TestEnded as e:
            print(f"[WARN] 测试被模拟器终止, 优雅收尾: {e}")
        # 收尾
        try:
            sim.exit()
            print("\n[EXIT] 已退出")
        except Exception as e:
            print(f"[EXIT] 失败(测试可能已结束): {e}")
        # 统计
        n = len(sim.cleared)
        vt = sim.vt
        print("\n========== 统计 ==========")
        print(f"清除干扰源数 : {n}")
        print(f"虚拟世界总时间: {vt:.1f} s ({vt/3600:.2f} h)")
        print(f"平均定位清除时间: {(vt/n if n else float('inf')):.1f} s/个")
        print(f"检测次数 {sim.measure_cnt}, 换频 {sim.switch_cnt} 次, 清除失败 {sim.clear_fail} 次")
        print(f"程序现实运行时间: {sim.elapsed_real():.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="参赛队号(robot_id)")
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--wait", type=int, default=600, help="等待测试窗口秒数")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    os.makedirs("logs", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = open(f"logs/robot_p{args.problem}_{stamp}.jsonl", "w", encoding="utf-8")
    print(f"robot_id={args.team}, 问题{args.problem}, 接口 {base}")
    print(f"日志: {log.name}")
    print("等待测试窗口开启...(请在模拟器中确认开始演练测试)")

    sim = Sim(base, args.team, log)
    sim.enter(wait_s=args.wait)
    Robot(sim, args.problem).run()


if __name__ == "__main__":
    main()
