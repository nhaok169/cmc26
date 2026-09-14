# -*- coding: utf-8 -*-
"""核对正文数字：闭式误差、D*、四准则网格。结果写入 q2/logs/derive_check.json。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "q1"))
from geometry import DELTA, diam_and_axis, expected_time, k_expr, mec, region, wedge_hp
from solver import bearing_deg, locate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(OUT, exist_ok=True)

K = 40.0 / (2.0 * np.tan(DELTA))
K_RAD = 40.0 / (2.0 * DELTA)
D_STAR = 20.0 / np.tan(DELTA)
C_LEVER = 0.5 / np.sqrt(0.75)
D_NEAR, D_FAR = 5.0, 1500.0
EPS3 = np.array([-DELTA, 0.0, DELTA])


def lever_h(t):
    return C_LEVER * max(abs(D_FAR - t), abs(t - D_NEAR))


def closed_diam(d1, d2, gamma):
    s, c = np.sin(gamma), np.abs(np.cos(gamma))
    if abs(s) < 1e-8:
        return np.inf
    return 2.0 * DELTA * np.sqrt((d1 * d1 + d2 * d2 + 2.0 * d1 * d2 * c) / (s * s))


def poly_diam(d1, d2, gamma):
    G = np.array([0.0, 0.0])
    S1 = np.array([-d1, 0.0])
    S2 = d2 * np.array([np.cos(np.pi - gamma), np.sin(np.pi - gamma)])
    th1 = np.arctan2(G[1] - S1[1], G[0] - S1[0])
    th2 = np.arctan2(G[1] - S2[1], G[0] - S2[0])
    v = region(wedge_hp(S1, th1) + wedge_hp(S2, th2))
    if len(v) < 3:
        return None, None, None
    d, _ = diam_and_axis(v)
    r, _ = mec(v)
    return d, r, v


def _closedform_pack(rec):
    rel = np.array([x[0] for x in rec])
    worst = rec[int(np.argmax(rel))]
    return {
        "n": int(len(rec)),
        "max_pct": float(rel.max() * 100),
        "mean_pct": float(rel.mean() * 100),
        "max_rel_pct": float(rel.max() * 100),
        "p95_rel_pct": float(np.percentile(rel, 95) * 100),
        "mean_rel_pct": float(rel.mean() * 100),
        "worst": {
            "rel_pct": float(worst[0] * 100),
            "dloc": float(worst[1]),
            "dpoly": float(worst[2]),
            "d1": float(worst[3]),
            "d2": float(worst[4]),
            "gamma": int(worst[5]),
            "gamma_deg": int(worst[5]),
        },
    }


def _closedform_grid(gmin, gmax):
    """d1×ratio grid with d2∈[80,2000]; drop empty/unbounded polygons."""
    rec = []
    n_geom = 0
    n_drop = 0
    for d1 in (300.0, 500.0, 800.0, 1100.0, 1400.0):
        for ratio in (0.4, 0.7, 1.0, 1.4, 1.8):
            d2 = d1 * ratio
            if d2 < 80 or d2 > 2000:
                continue
            for gdeg in range(int(gmin), int(gmax) + 1, 5):
                n_geom += 1
                g = np.deg2rad(gdeg)
                dloc = closed_diam(d1, d2, g)
                dpoly, r, _ = poly_diam(d1, d2, g)
                if dpoly is None or not np.isfinite(dloc) or dpoly < 1:
                    n_drop += 1
                    continue
                rec.append((abs(dloc - dpoly) / dpoly, dloc, dpoly, d1, d2, gdeg, r))
    out = _closedform_pack(rec)
    out["n_geom"] = int(n_geom)
    out["n_drop_empty_unbounded"] = int(n_drop)
    out["gamma_deg"] = [int(gmin), int(gmax), 5]
    return out


def closedform_error():
    """Two protocols: working [30°,150°] and widened [25°,155°], step 5°."""
    work = _closedform_grid(30, 150)
    wide = _closedform_grid(25, 155)
    return {"work": work, "wide": wide}


def equal_range_800():
    d = 800.0
    out = {}
    for gdeg in (30, 90, 150):
        g = np.deg2rad(gdeg)
        dloc = closed_diam(d, d, g)
        dpoly, r, _ = poly_diam(d, d, g)
        out[str(gdeg)] = {"dloc": float(dloc), "dpoly": float(dpoly), "mec": float(r)}
    out["inf_equal"] = float(2.0 * np.sqrt(2.0) * DELTA * d)
    out["inf_abs"] = float(2.0 * DELTA * d)
    out["inf_tan"] = float(2.0 * d * np.tan(DELTA))
    out["D_star"] = float(D_STAR)
    out["K_tan"] = float(K)
    out["K_rad"] = float(K_RAD)
    # D such that equal-range 90° hits 40 m
    out["D_equal_40"] = float(40.0 / (2.0 * np.sqrt(2.0) * DELTA))
    return out


def k_ok(t, h, D):
    return max(k_expr(t, h, D, e) for e in EPS3) <= (K * K)


def k_ok9(t, h, D):
    """K-condition worst over e1,e2 ∈ {-δ,0,δ}. True geometry only depends on e1 (G offset)."""
    return max(k_expr(t, h, D, e1) for e1 in EPS3 for _e2 in EPS3) <= (K * K)


def two_station_mec(t, h, D, e1, e2):
    """Measured axis = x-axis; true G = D (cos e1, sin e1); S2 bearing error e2."""
    S1 = np.array([0.0, 0.0])
    S2 = np.array([float(t), float(h)])
    G = np.array([D * np.cos(e1), D * np.sin(e1)])
    th1 = 0.0
    th2 = bearing_deg(S2, G) + float(np.degrees(e2))
    loc = locate([S1, S2], [th1, th2], 1.0)
    if loc.status != "OK" or loc.mec_radius is None or not np.isfinite(loc.mec_radius):
        return float("inf")
    return float(loc.mec_radius)


def worst_poly_mec(t, h, D):
    worst = -1.0
    worst_e = (0.0, 0.0)
    for e1 in EPS3:
        for e2 in EPS3:
            r = two_station_mec(t, h, D, e1, e2)
            if r > worst:
                worst = r
                worst_e = (float(np.degrees(e1)), float(np.degrees(e2)))
    return float(worst), worst_e


def verify_minimax(t=550.0, h=450.0):
    """Paper: K-prefix ≈724 m; D∈[5,730] step 2.5, 9 endpoints, worst R_MEC≈19.94 m; first >20 at ≈735 m."""
    Ds = np.linspace(5.0, 1500.0, 80)
    mask = np.array([k_ok9(t, h, float(D)) for D in Ds])
    if mask[0]:
        k = 0
        while k < len(mask) and mask[k]:
            k += 1
        d_prefix = float(Ds[k - 1]) if k else 0.0
    else:
        d_prefix = 0.0

    D_fine = np.arange(5.0, 900.0 + 1e-9, 2.5)
    worst_r = -1.0
    worst_D = None
    worst_e = None
    n_in730 = 0
    first_viol = None
    for D in D_fine:
        r, e = worst_poly_mec(t, h, float(D))
        if D <= 730.0 + 1e-9:
            n_in730 += 1
            if r > worst_r:
                worst_r, worst_D, worst_e = r, float(D), e
        if first_viol is None and r > 20.0 + 1e-9:
            first_viol = {
                "D": float(D),
                "R_MEC": float(r),
                "e1_deg": float(e[0]),
                "e2_deg": float(e[1]),
            }

    return {
        "t": float(t),
        "h": float(h),
        "k_prefix_m": d_prefix,
        "n_error_endpoints": 9,
        "poly_D_lo": 5.0,
        "poly_D_hi": 730.0,
        "poly_step_m": 2.5,
        "n_D_in_730": int(n_in730),
        "worst_R_MEC": float(worst_r),
        "worst_D": worst_D,
        "worst_e_deg": {"e1": None if worst_e is None else worst_e[0], "e2": None if worst_e is None else worst_e[1]},
        "first_violation": first_viol,
        "move_m": float(np.hypot(t, h)),
    }


def two_step_interval(t, h, Ds):
    m = np.array([k_ok(t, h, D) for D in Ds])
    idx = np.where(m)[0]
    if len(idx) == 0:
        return None, 0.0, m
    return (float(Ds[idx[0]]), float(Ds[idx[-1]])), float(m.mean()), m


def school_grid():
    ts = np.arange(250.0, 1001.0, 25.0)
    hs = np.arange(150.0, 701.0, 25.0)
    Ds = np.linspace(5.0, 1500.0, 80)
    w = Ds.copy()
    w = w / w.sum()

    rows = []
    for t in ts:
        for h in hs:
            if abs(h) < 40:
                continue
            iv, pfrac, mask = two_step_interval(t, h, Ds)
            # minimax score: longest prefix from D≈5
            if mask[0]:
                k = 0
                while k < len(mask) and mask[k]:
                    k += 1
                d_prefix = float(Ds[k - 1]) if k else 0.0
            else:
                d_prefix = 0.0
            p_area = float(w[mask].sum()) if mask.any() else 0.0
            lev_ok = abs(h) + 1e-9 >= lever_h(t) - 1.0
            rows.append(
                {
                    "t": float(t),
                    "h": float(h),
                    "interval": iv,
                    "p_area": p_area,
                    "d_prefix": d_prefix,
                    "lever_ok": bool(lev_ok),
                    "move": float(np.hypot(t, h)),
                }
            )

    best_mm = max(rows, key=lambda r: (r["d_prefix"], -r["move"]))
    best_p = max(rows, key=lambda r: (r["p_area"], -r["move"]))

    # E[T] on coarser grid
    Ds_et = np.linspace(5.0, 1500.0, 40)
    w_et = Ds_et.copy()
    w_et /= w_et.sum()
    rho = lambda L: max(50.0, 0.5 * L)
    tt = np.arange(350.0, 951.0, 50.0)
    hh = np.arange(200.0, 601.0, 50.0)
    et_rows = []
    for t in tt:
        for h in hh:
            et, p, en = expected_time(t, h, Ds_et, w=w_et, rho_rule=rho)
            et_rows.append(
                {
                    "t": float(t),
                    "h": float(h),
                    "et": float(et),
                    "p": float(p),
                    "en": float(en),
                    "lever_ok": bool(abs(h) >= lever_h(t) - 1.0),
                }
            )
    finite = [r for r in et_rows if np.isfinite(r["et"])]
    best_time = min(finite, key=lambda r: r["et"])
    cons = [r for r in finite if r["lever_ok"]]
    best_cons = min(cons, key=lambda r: r["et"]) if cons else None

    named = []
    for t, h, name in (
        (550.0, 450.0, "minimax_paper"),
        (675.0, 500.0, "prob_paper"),
        (750.0, 450.0, "cons_paper"),
        (500.0, 250.0, "time_paper"),
    ):
        et, p, en = expected_time(t, h, Ds_et, w=w_et, rho_rule=rho)
        iv, _, _ = two_step_interval(t, h, Ds)
        named.append(
            {
                "name": name,
                "t": t,
                "h": h,
                "et": float(et),
                "p": float(p),
                "en": float(en),
                "interval": iv,
                "lever_ok": bool(abs(h) >= lever_h(t) - 1.0),
            }
        )

    et_time = next(r for r in named if r["name"] == "time_paper")["et"]
    et_cons = next(r for r in named if r["name"] == "cons_paper")["et"]
    return {
        "grid_th": {"t": [float(x) for x in ts], "h": [float(x) for x in hs], "n": len(rows)},
        "best_minimax_prefix": best_mm,
        "best_prob": best_p,
        "et_grid": {"t": [float(x) for x in tt], "h": [float(x) for x in hh], "n": len(et_rows)},
        "best_time": best_time,
        "best_cons": best_cons,
        "named": named,
        "speedup_time_vs_cons": float((et_cons - et_time) / et_cons) if et_cons else None,
    }


def main():
    err = closedform_error()
    eq = equal_range_800()
    print("closedform_error.work", json.dumps(err["work"], ensure_ascii=False, default=str))
    print("closedform_error.wide", json.dumps(err["wide"], ensure_ascii=False, default=str))
    print(
        "paper targets: work n=600 max≈0.434% mean≈0.086% worst(500,500,30°); "
        "wide n=648 max≈0.63% mean≈0.10% worst(1100,1100,25°)"
    )
    print("equal-range 800", eq)
    mm = verify_minimax()
    print("verify_minimax", json.dumps(mm, ensure_ascii=False, default=str))
    print("paper targets: prefix≈724 m, worst R_MEC≈19.94 m on [5,730], first violation≈735 m")
    schools = school_grid()
    print("schools named", json.dumps(schools["named"], ensure_ascii=False, indent=2, default=str))
    payload = {
        "closedform_error": err,
        "closedform": err,
        "equal800": eq,
        "verify_minimax": mm,
        "schools": schools,
        "K": K,
        "D_star": D_STAR,
    }
    path = os.path.join(OUT, "derive_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
