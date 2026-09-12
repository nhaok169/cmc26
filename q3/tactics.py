# -*- coding: utf-8 -*-
"""问题3/4 共用：就近追击、近侧第二点、当前位置顺便测。"""
from __future__ import annotations

import math

import numpy as np


def unit(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([math.cos(t), math.sin(t)], dtype=float)


def clamp_arena(p, radius: float = 1790.0) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = float(np.linalg.norm(p))
    if n > radius and n > 1e-9:
        p = p * (radius / n)
    return p


def estimate_pos(hits, dist: float = 480.0) -> np.ndarray:
    S, th = hits[-1]
    return clamp_arena(np.asarray(S, float) + dist * unit(th))


def closer_offset(S, theta_deg, robot, t: float, h: float) -> np.ndarray:
    """(t, ±h) 里离当前机器狗更近的那一个。"""
    S = np.asarray(S, float)
    e = unit(theta_deg)
    n = np.array([-e[1], e[0]])
    p1 = clamp_arena(S + t * e + h * n)
    p2 = clamp_arena(S + t * e - h * n)
    r = np.asarray(robot, float)
    if float(np.linalg.norm(p1 - r)) <= float(np.linalg.norm(p2 - r)):
        return p1
    return p2


def ray_coords(S, theta_deg, p):
    """相对第一站示向度的纵向距离与侧向间隔。"""
    e = unit(theta_deg)
    v = np.asarray(p, float) - np.asarray(S, float)
    t_along = float(e @ v)
    hlat = abs(float(-e[1] * v[0] + e[0] * v[1]))
    return t_along, hlat


def lateral_offset(S, theta_deg, p) -> float:
    return ray_coords(S, theta_deg, p)[1]


def ang_diff_deg(a, b) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def greedy_order(pts, start) -> list:
    left = [np.asarray(p, float) for p in pts]
    cur = np.asarray(start, float)
    order = []
    while left:
        k = min(range(len(left)), key=lambda i: float(np.linalg.norm(left[i] - cur)))
        order.append(left.pop(k))
        cur = order[-1]
    return order


def diverse_order(pts, visited) -> list:
    """每次取离已访问点最远的，把监听点铺开，便于早停。"""
    pool = [np.asarray(p, float) for p in pts]
    refs = [np.asarray(v, float) for v in visited]
    order = []
    if not refs:
        refs = [np.zeros(2)]
    while pool:
        def score(p):
            return min(float(np.linalg.norm(p - q)) for q in refs)

        k = max(range(len(pool)), key=lambda i: score(pool[i]))
        p = pool.pop(k)
        order.append(p)
        refs.append(p)
    return order
