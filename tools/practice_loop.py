# -*- coding: utf-8 -*-
"""官方模拟器多次演练：登录后停在「演练测试」页，由脚本点开始、跑机器狗、点返回。

用法（先打开模拟器并登录）：
  python tools/practice_loop.py --problem 3 --n 10
  python tools/practice_loop.py --problem 4 --n 8
  python tools/practice_loop.py --probe          # 列出窗口里能点到的按钮名
  python tools/practice_loop.py --problem 3 --n 10 --no-click
      # 不点界面：你每局点「开始」，脚本自动 /enter 并跑完；你再点「返回」后点下一局

只等你点开始、不点界面也可以：
  python q3/run_q3.py --watch
  python q4/run_q4.py --watch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "q1"))
sys.path.insert(0, str(ROOT / "q2"))
sys.path.insert(0, str(ROOT / "q3"))
sys.path.insert(0, str(ROOT / "q4"))

WIN_HINTS = ("环境模拟器", "无线电干扰源")
START_BTN = {
    3: "开始问题3演练测试",
    4: "开始问题4演练测试",
}
RETURN_BTNS = ("返回演练测试", "返回演练", "返回")
OFFICIAL_DIR = ROOT / "simulator" / "JammersSimulatorData" / "behavior-logs"


def _walk(ctrl, max_depth=14, depth=0):
    yield ctrl
    if depth >= max_depth:
        return
    try:
        kids = ctrl.GetChildren()
    except Exception:
        return
    for c in kids:
        yield from _walk(c, max_depth, depth + 1)


def find_sim_window():
    import uiautomation as auto

    root = auto.GetRootControl()
    for w in root.GetChildren():
        name = w.Name or ""
        if any(h in name for h in WIN_HINTS) and "DeepSeek" not in name and "Cursor" not in name:
            return w
    return None


def list_named_controls(win) -> list:
    rows = []
    for c in _walk(win):
        name = (c.Name or "").strip()
        if not name:
            continue
        rows.append((c.ControlTypeName, name))
    return rows


def click_named(win, names, timeout: float = 8.0) -> str | None:
    if isinstance(names, str):
        names = (names,)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            win.SetActive()
        except Exception:
            pass
        for c in _walk(win):
            name = (c.Name or "").strip()
            if not name:
                continue
            if not any(key in name for key in names):
                continue
            try:
                c.Click()
                return name
            except Exception:
                try:
                    pat = c.GetInvokePattern()
                    if pat:
                        pat.Invoke()
                        return name
                except Exception:
                    continue
        time.sleep(0.35)
        win = find_sim_window() or win
    return None


def latest_official(problem: int):
    files = sorted(
        OFFICIAL_DIR.glob(f"practice-p{problem}-*.result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def wait_enter(sim, timeout: float = 25.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            body = sim._post(
                "/enter",
                {
                    "arena_id": "default",
                    "robot_id": sim.robot_id,
                    "request_id": sim._new_id("enter"),
                },
                retries=1,
            )
            if body.get("accepted") is True:
                return body
            last = body
        except Exception as e:
            last = e
        time.sleep(0.6)
    raise RuntimeError(f"/enter 超时: {last}")


def run_one(problem: int, click: bool) -> dict:
    if problem == 3:
        from api import Simulator
        from run_q3 import Dog, ROBOT_ID, save_log
    else:
        from run_q4 import Dog, ROBOT_ID, Sim as Simulator, save_log

    win = find_sim_window() if click else None
    if click:
        if win is None:
            raise RuntimeError("找不到模拟器窗口。请先打开并登录，停在「演练测试」页。")
        hit = click_named(win, START_BTN[problem], timeout=6.0)
        if hit:
            print(f"clicked [{hit}]")
        else:
            print(f"warning: 没点到「{START_BTN[problem]}」，尝试直接 /enter")

    sim = Simulator(ROBOT_ID)
    body = wait_enter(sim, timeout=25.0)
    remain = float(body.get("remaining_real_duration_s", 1200))
    print(f"entered remaining={remain:.0f}s")
    dog = Dog(sim, remain_s=remain)
    try:
        summary = dog.run(do_enter=False)
    except Exception as e:
        print("run error:", e)
        try:
            sim.exit()
        except Exception:
            pass
        summary = {
            "error": str(e),
            "cleared": sorted(getattr(dog.book, "cleared", [])),
            "n_cleared": len(getattr(dog.book, "cleared", [])),
            "virtual_time_s": getattr(sim, "virtual_time_s", 0.0),
            "avg_s": None,
        }
    save_log(sim, summary, f"q{problem}")

    if click:
        time.sleep(0.6)
        win = find_sim_window() or win
        back = click_named(win, RETURN_BTNS, timeout=8.0)
        if back:
            print(f"clicked [{back}]")
        else:
            print("warning: 没点到「返回演练测试」，请手动点返回后再继续")
        time.sleep(0.8)

    official = latest_official(problem)
    n_true = official.get("jammer_count") if official else None
    n_cleared = int(summary.get("n_cleared") or 0)
    avg = summary.get("avg_s")
    if avg is None and n_cleared:
        avg = float(summary.get("virtual_time_s") or 0) / n_cleared
    ok = (n_true is not None) and (n_cleared >= n_true) and not summary.get("failed")
    row = {
        "case": official.get("case_code") if official else None,
        "n_true": n_true,
        "omni": official.get("omnidirectional_jammer_count") if official else None,
        "dir": official.get("directional_jammer_count") if official else None,
        "n_cleared": n_cleared,
        "avg_s": avg,
        "virtual_s": summary.get("virtual_time_s"),
        "ok": ok,
        "failed": summary.get("failed") or [],
    }
    return row


def probe():
    win = find_sim_window()
    if win is None:
        print("找不到模拟器窗口（标题需含「环境模拟器」）。")
        return 1
    print("window:", win.Name)
    print("named controls:")
    seen = set()
    for typ, name in list_named_controls(win):
        key = (typ, name)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {typ:24s} {name}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, choices=(3, 4), default=3)
    ap.add_argument("--n", type=int, default=5, help="连续演练局数")
    ap.add_argument("--no-click", action="store_true", help="不点界面，只在你点开始后自动跑")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        sys.exit(probe())

    click = not args.no_click
    print(
        f"practice loop problem={args.problem} n={args.n} click={click}  "
        f"Ctrl+C to stop"
    )
    if click:
        print("请保持模拟器在「演练测试」页、不要最小化。登录只需做一次。")
    else:
        print("每局请先点「开始问题X演练测试」；跑完后点「返回演练测试」再点下一局。")

    rows = []
    try:
        for i in range(args.n):
            print(f"\n======== game {i+1}/{args.n} ========")
            if (not click) and i > 0:
                print("waiting for you to start the next test...")
            row = run_one(args.problem, click=click)
            rows.append(row)
            flag = "OK" if row["ok"] else "CHECK"
            avg = f"{row['avg_s']:.0f}" if row["avg_s"] else "?"
            print(
                f"-> {flag} case={row['case']} "
                f"cleared={row['n_cleared']}/{row['n_true']} "
                f"(omni={row['omni']} dir={row['dir']}) avg={avg}s"
            )
            if not click:
                time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nstopped by user")

    if not rows:
        return
    print("\n======== summary ========")
    oks = sum(1 for r in rows if r["ok"])
    avgs = [r["avg_s"] for r in rows if r["avg_s"]]
    for i, r in enumerate(rows, 1):
        flag = "OK" if r["ok"] else "CHECK"
        avg = f"{r['avg_s']:.0f}" if r["avg_s"] else "?"
        print(
            f"{i:02d} {flag} {r['case']}  "
            f"{r['n_cleared']}/{r['n_true']}  avg={avg}s  failed={r['failed']}"
        )
    mean = sum(avgs) / len(avgs) if avgs else None
    print(f"full-clear {oks}/{len(rows)}  mean avg={mean:.0f}s" if mean else f"full-clear {oks}/{len(rows)}")


if __name__ == "__main__":
    main()
