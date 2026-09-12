"""
Problem B, Question 1: bearing-only intersection, set diameter, and coverage test.

The script implements the revised Question 1 model:

1. Each direction measurement with an error bound [-1 deg, +1 deg] defines an
   angular wedge.  The feasible source region is the intersection of these
   wedges (a convex polygon if bounded).
2. The region diameter is the Euclidean set diameter, obtained on the convex
   hull vertices by the rotating-calipers algorithm.
3. Whether a circle with radius D/2 covers the region is not automatic.  It is
   equivalent to r_MEC == D/2, where r_MEC is the radius of the minimum
   enclosing circle (Welzl's algorithm).

Outputs are written next to this script:
  - q1_vertices.csv
  - q1_summary.csv
  - q1_monte_carlo.csv
  - fig_q1_geometry.pdf / .png / .svg
  - fig_q1_criterion_cases.pdf / .png / .svg
  - fig_q1_robustness.pdf / .png / .svg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parent
EPS = 1e-9
DEG = math.pi / 180.0
TARGET_RADIUS = 1800.0


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Times New Roman"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
    }
)


@dataclass(frozen=True)
class Circle:
    """A circle represented by its center and radius."""

    center: np.ndarray
    radius: float


@dataclass
class RegionResult:
    """Result of intersecting all bearing wedges."""

    polygon: np.ndarray
    hull: np.ndarray
    diameter: float
    diameter_pair: tuple[np.ndarray, np.ndarray]
    diameter_circle: Circle
    mec: Circle
    covered: bool
    bounded_by_angle_only: bool


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Return the scalar z-component of a x b."""

    return float(a[0] * b[1] - a[1] * b[0])


def unit_vector(angle_rad: float) -> np.ndarray:
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def bearing_from_station(station: Sequence[float], target: Sequence[float]) -> float:
    """Return a bearing in [0, 360) degrees from station to target."""

    sx, sy = station
    tx, ty = target
    deg = math.degrees(math.atan2(ty - sy, tx - sx))
    return deg % 360.0


def disk_polygon(radius: float = TARGET_RADIUS, n: int = 720) -> np.ndarray:
    """Approximate the target disk by a regular n-gon."""

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def halfplanes_for_wedge(station: np.ndarray, bearing_deg: float, delta_deg: float) -> list[tuple[float, float, float]]:
    """Return the two half-planes a*x + b*y <= c for one error wedge.

    The wedge is the angular sector centered at `bearing_deg` with half-angle
    `delta_deg`.  The returned inequalities are ordered counter-clockwise from
    the lower to the upper boundary.
    """

    theta = math.radians(bearing_deg)
    delta = math.radians(delta_deg)
    u_low = unit_vector(theta - delta)
    u_high = unit_vector(theta + delta)
    sx, sy = float(station[0]), float(station[1])

    # Lower boundary: cross(u_low, P - S) >= 0.
    lower = (
        u_low[1],
        -u_low[0],
        u_low[1] * sx - u_low[0] * sy,
    )

    # Upper boundary: cross(P - S, u_high) >= 0.
    upper = (
        -u_high[1],
        u_high[0],
        u_high[0] * sy - u_high[1] * sx,
    )
    return [lower, upper]


def clip_polygon_by_halfplane(
    polygon: np.ndarray, halfplane: tuple[float, float, float], tol: float = EPS
) -> np.ndarray:
    """Clip a convex polygon by a*x + b*y <= c (Sutherland-Hodgman)."""

    a, b, c = halfplane
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=float)

    out: list[np.ndarray] = []
    n = len(polygon)
    for i in range(n):
        p = polygon[i]
        q = polygon[(i + 1) % n]
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c

        if fp <= tol:
            out.append(p.copy())

        if (fp > tol and fq < -tol) or (fp < -tol and fq > tol):
            t = fp / (fp - fq)
            out.append(p + t * (q - p))

    if not out:
        return np.empty((0, 2), dtype=float)
    return np.asarray(out, dtype=float)


def orient_ccw(polygon: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""

    if len(polygon) < 3:
        return polygon
    x = polygon[:, 0]
    y = polygon[:, 1]
    area2 = float(
        np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    )
    return polygon[::-1].copy() if area2 < 0 else polygon.copy()


def remove_collinear_points(polygon: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Remove nearly collinear points from a convex polygon."""

    if len(polygon) <= 3:
        return polygon.copy()
    pts = orient_ccw(polygon)
    out: list[np.ndarray] = []
    n = len(pts)
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        area2 = abs(cross_2d(b - a, c - b))
        if area2 > tol:
            out.append(b)
    return np.asarray(out, dtype=float) if out else pts.copy()


def convex_hull_ccw(points: np.ndarray) -> np.ndarray:
    """Return convex hull vertices in counter-clockwise order."""

    if len(points) < 3:
        return points.copy()
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        return unique
    hull = ConvexHull(unique)
    vertices = unique[hull.vertices]
    return remove_collinear_points(orient_ccw(vertices))


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(
        float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    )


def brute_force_diameter(points: np.ndarray) -> tuple[float, int, int]:
    """Return max squared distance and its pair of indices."""

    best = -1.0
    pair = (0, 1)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = float(np.dot(points[i] - points[j], points[i] - points[j]))
            if d2 > best:
                best = d2
                pair = (i, j)
    return best, pair[0], pair[1]


def rotating_calipers_diameter(points: np.ndarray) -> tuple[float, int, int]:
    """Compute the diameter of a convex polygon by rotating calipers."""

    n = len(points)
    if n == 2:
        d2 = float(np.dot(points[0] - points[1], points[0] - points[1]))
        return d2, 0, 1
    if n < 2:
        raise ValueError("At least two points are required.")

    j = 1
    while True:
        nj = (j + 1) % n
        current = abs(cross_2d(points[1] - points[0], points[nj] - points[0]))
        nxt = abs(cross_2d(points[1] - points[0], points[j] - points[0]))
        if current > nxt:
            j = nj
        else:
            break

    best = -1.0
    pair = (0, 1)
    for i in range(n):
        while True:
            nj = (j + 1) % n
            current = cross_2d(points[(i + 1) % n] - points[i], points[nj] - points[i])
            nxt = cross_2d(points[(i + 1) % n] - points[i], points[j] - points[i])
            if current > nxt:
                j = nj
            else:
                break

        for k in (j, (j + 1) % n):
            d2 = float(np.dot(points[i] - points[k], points[i] - points[k]))
            if d2 > best:
                best = d2
                pair = (i, k)

    return best, pair[0], pair[1]


def circle_from_one(points: list[np.ndarray]) -> Circle:
    p = points[0]
    return Circle(p.copy(), 0.0)


def circle_from_two(a: np.ndarray, b: np.ndarray) -> Circle:
    center = 0.5 * (a + b)
    radius = 0.5 * float(np.linalg.norm(a - b))
    return Circle(center, radius)


def circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Circle:
    """Circumcircle of three points; fall back to a 2-point circle if collinear."""

    d = 2.0 * cross_2d(b - a, c - a)
    if abs(d) < 1e-12:
        candidates = [circle_from_two(a, b), circle_from_two(a, c), circle_from_two(b, c)]
        return max(candidates, key=lambda circle: circle.radius)

    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    cc = float(np.dot(c, c))
    ux = (
        (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1]))
        / d
    )
    uy = (
        (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0]))
        / d
    )
    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(center - a))
    return Circle(center, radius)


def circle_from_points(points: list[np.ndarray]) -> Circle:
    n = len(points)
    if n == 0:
        return Circle(np.zeros(2), 0.0)
    if n == 1:
        return circle_from_one(points)
    if n == 2:
        return circle_from_two(points[0], points[1])
    if n == 3:
        return circle_from_three(points[0], points[1], points[2])
    raise ValueError("A minimum enclosing circle is determined by at most 3 points.")


def welzl_minimum_enclosing_circle(points: Iterable[Sequence[float]]) -> Circle:
    """Return the minimum enclosing circle by Welzl's randomized algorithm."""

    p = [np.asarray(x, dtype=float).copy() for x in points]
    # Deterministic shuffling keeps figure output reproducible.
    rng = np.random.default_rng(20260911)
    rng.shuffle(p)

    def recurse(remaining: list[np.ndarray], boundary: list[np.ndarray]) -> Circle:
        if not remaining or len(boundary) == 3:
            return circle_from_points(boundary)
        pt = remaining.pop()
        circle = recurse(remaining, boundary)
        if np.linalg.norm(pt - circle.center) <= circle.radius + 1e-8:
            remaining.append(pt)
            return circle
        boundary.append(pt)
        circle = recurse(remaining, boundary)
        boundary.pop()
        remaining.append(pt)
        return circle

    return recurse(p, [])


def intersect_bearing_wedges(
    stations: Sequence[Sequence[float]],
    bearings_deg: Sequence[float],
    delta_deg: float = 1.0,
    domain_radius: float = TARGET_RADIUS,
    domain_n: int = 720,
) -> tuple[np.ndarray, bool]:
    """Intersect all bearing wedges and return (polygon, angle_only_bounded).

    The target disk is used as a physically known compact domain.  If the
    angle-only intersection touches the disk boundary, the result is
    considered 'bounded by the prior domain' rather than by the angular
    measurements alone.
    """

    polygon = disk_polygon(radius=domain_radius, n=domain_n)
    for station, bearing in zip(stations, bearings_deg):
        for halfplane in halfplanes_for_wedge(np.asarray(station, dtype=float), bearing, delta_deg):
            polygon = clip_polygon_by_halfplane(polygon, halfplane)
            if len(polygon) < 3:
                return np.empty((0, 2), dtype=float), False

    hull = convex_hull_ccw(polygon)
    if len(hull) < 3:
        return hull, False

    angle_only_bounded = not np.any(
        np.abs(np.linalg.norm(hull, axis=1) - domain_radius) < 0.01
    )
    return hull, angle_only_bounded


def analyze_region(polygon: np.ndarray) -> RegionResult:
    """Analyze a convex polygon: diameter, diameter circle, and MEC coverage."""

    hull = convex_hull_ccw(polygon)
    if len(hull) < 2:
        raise ValueError("The feasible region is empty or degenerate.")

    d2_caliper, ia, ib = rotating_calipers_diameter(hull)
    d2_brute, ja, jb = brute_force_diameter(hull)
    if not math.isclose(d2_caliper, d2_brute, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            "Rotating-calipers diameter disagrees with brute force: "
            f"{d2_caliper:.12e} vs {d2_brute:.12e}"
        )

    diameter = math.sqrt(d2_caliper)
    p_star = hull[ia]
    q_star = hull[ib]
    diameter_circle = circle_from_two(p_star, q_star)
    mec = welzl_minimum_enclosing_circle(hull)

    covered = all(
        np.linalg.norm(v - diameter_circle.center) <= diameter_circle.radius + 1e-8
        for v in hull
    )
    # Numerical consistency check between the direct Thales test and the MEC.
    if covered and mec.radius > 0.5 * diameter + 1e-7:
        raise RuntimeError("Inconsistent coverage: Thales circle covers, but MEC is larger.")

    return RegionResult(
        polygon=polygon,
        hull=hull,
        diameter=diameter,
        diameter_pair=(p_star, q_star),
        diameter_circle=diameter_circle,
        mec=mec,
        covered=covered,
        bounded_by_angle_only=True,
    )


def make_synthetic_case() -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Create a bounded four-station bearing scenario for the main figure."""

    true_source = np.array([125.0, -65.0])
    stations = []
    for angle_deg, distance in zip([35.0, 130.0, 220.0, 305.0], [980.0, 1180.0, 910.0, 1050.0]):
        angle = math.radians(angle_deg)
        station = true_source + distance * np.array([math.cos(angle), math.sin(angle)])
        # Keep the synthetic stations inside the target disk.
        if np.linalg.norm(station) > TARGET_RADIUS - 10:
            station *= (TARGET_RADIUS - 10) / np.linalg.norm(station)
        stations.append(station)

    errors = [-0.72, 0.41, -0.35, 0.68]
    bearings = [
        bearing_from_station(station, true_source) + error
        for station, error in zip(stations, errors)
    ]
    return true_source, stations, bearings


def draw_region_analysis(
    ax: plt.Axes,
    stations: Sequence[np.ndarray],
    bearings_deg: Sequence[float],
    true_source: np.ndarray | None,
    result: RegionResult,
    ray_length: float = 2600.0,
    domain_radius: float = TARGET_RADIUS,
) -> None:
    """Draw one complete Question 1 geometric analysis."""

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        domain_radius * np.cos(theta),
        domain_radius * np.sin(theta),
        color="0.65",
        linewidth=0.9,
        linestyle="--",
        label="目标区域边界",
    )

    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    for idx, (station, bearing) in enumerate(zip(stations, bearings_deg)):
        station = np.asarray(station, dtype=float)
        low = math.radians(bearing - 1.0)
        high = math.radians(bearing + 1.0)
        for angle in (low, high):
            end = station + ray_length * unit_vector(angle)
            ax.plot(
                [station[0], end[0]],
                [station[1], end[1]],
                color=colors[idx % len(colors)],
                linewidth=0.9,
                alpha=0.55,
            )
        ax.scatter(
            station[0],
            station[1],
            s=48,
            color=colors[idx % len(colors)],
            marker="s",
            zorder=4,
            label=f"检测点 {idx + 1}",
        )

    hull = result.hull
    ax.fill(
        np.r_[hull[:, 0], hull[0, 0]],
        np.r_[hull[:, 1], hull[0, 1]],
        color="#377eb8",
        alpha=0.22,
        label="定位区域",
    )
    ax.plot(
        np.r_[hull[:, 0], hull[0, 0]],
        np.r_[hull[:, 1], hull[0, 1]],
        color="#377eb8",
        linewidth=1.7,
        label="凸多边形边界",
    )

    p, q = result.diameter_pair
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#e41a1c", linewidth=1.8, label="直径线段")

    dc = result.diameter_circle
    dc_theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        dc.center[0] + dc.radius * np.cos(dc_theta),
        dc.center[1] + dc.radius * np.sin(dc_theta),
        color="#e41a1c",
        linewidth=1.4,
        linestyle="--",
        label=f"直径圆, r=D/2={dc.radius:.3f}",
    )

    mec = result.mec
    ax.plot(
        mec.center[0] + mec.radius * np.cos(dc_theta),
        mec.center[1] + mec.radius * np.sin(dc_theta),
        color="#ff7f00",
        linewidth=1.4,
        linestyle=":",
        label=f"最小包围圆, r={mec.radius:.3f}",
    )

    if true_source is not None:
        ax.scatter(
            true_source[0],
            true_source[1],
            s=90,
            marker="*",
            color="#000000",
            zorder=5,
            label="真实干扰源",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")


def draw_criterion_cases() -> None:
    """Illustrate when a diameter circle can and cannot cover a convex set."""

    cases = [
        ("等边三角形：不能覆盖", np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.464101615]])),
        ("锐角三角形：不能覆盖", np.array([[0.0, 0.0], [5.0, 0.0], [2.0, 4.0]])),
        ("直角三角形：可以覆盖", np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])),
        ("矩形：可以覆盖", np.array([[-2.0, -1.5], [2.0, -1.5], [2.0, 1.5], [-2.0, 1.5]])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    for ax, (title, polygon) in zip(axes.ravel(), cases):
        result = analyze_region(polygon)
        draw_region_analysis(ax, [], [], None, result, domain_radius=5.0)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - 0.25, xlim[1] + 0.25)
        ax.set_ylim(ylim[0] - 0.25, ylim[1] + 0.25)
        ax.set_title(
            f"{title}\nD={result.diameter:.4f}, D/2={0.5*result.diameter:.4f}, "
            f"r_MEC={result.mec.radius:.4f}",
            fontsize=10,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("直径圆覆盖判据的典型情形", y=1.02)
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"fig_q1_criterion_cases.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo(n_reps_per_n: int = 500, seed: int = 20260911) -> pd.DataFrame:
    """Monte Carlo robustness study over measurement errors and station counts."""

    rng = np.random.default_rng(seed)
    true_source = np.array([0.0, 0.0])
    rows: list[dict[str, float | int | bool]] = []

    for n_stations in range(2, 7):
        for _ in range(n_reps_per_n):
            base_angles = np.linspace(0.0, 360.0, n_stations, endpoint=False) + rng.uniform(
                0.0, 40.0
            )
            distances = rng.uniform(700.0, 1400.0, size=n_stations)
            stations = []
            for angle_deg, distance in zip(base_angles, distances):
                angle = math.radians(angle_deg)
                station = true_source + distance * np.array([math.cos(angle), math.sin(angle)])
                if np.linalg.norm(station) > TARGET_RADIUS - 10:
                    station *= (TARGET_RADIUS - 10) / np.linalg.norm(station)
                stations.append(station)

            true_bearings = [
                bearing_from_station(station, true_source) for station in stations
            ]
            errors = rng.uniform(-1.0, 1.0, size=n_stations)
            measured_bearings = [
                (theta + eps) % 360.0 for theta, eps in zip(true_bearings, errors)
            ]

            polygon, bounded = intersect_bearing_wedges(stations, measured_bearings)
            if len(polygon) < 3 or not bounded:
                continue

            result = analyze_region(polygon)
            rows.append(
                {
                    "n_stations": n_stations,
                    "D": result.diameter,
                    "r_mec": result.mec.radius,
                    "D_over_2": 0.5 * result.diameter,
                    "kappa": result.mec.radius / (0.5 * result.diameter),
                    "covered": result.covered,
                    "area": polygon_area(result.hull),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "q1_monte_carlo.csv", index=False, encoding="utf-8-sig")
    return df


def draw_robustness(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    n_values = sorted(df["n_stations"].unique())
    box_data = [df.loc[df["n_stations"] == n, "D"].to_numpy() for n in n_values]
    axes[0, 0].boxplot(box_data, tick_labels=[str(n) for n in n_values], showfliers=False)
    axes[0, 0].set_title("定位区域直径 D 的误差敏感性")
    axes[0, 0].set_xlabel("检测点数量 n")
    axes[0, 0].set_ylabel("D / m")
    axes[0, 0].grid(True, alpha=0.25)

    kappa_data = [df.loc[df["n_stations"] == n, "kappa"].to_numpy() for n in n_values]
    axes[0, 1].boxplot(kappa_data, tick_labels=[str(n) for n in n_values], showfliers=False)
    axes[0, 1].axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="r_MEC = D/2")
    axes[0, 1].set_title("最小包围圆半径与 D/2 的比值")
    axes[0, 1].set_xlabel("检测点数量 n")
    axes[0, 1].set_ylabel(r"$\kappa = r_{MEC}/(D/2)$")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    coverage = (
        df.groupby("n_stations")["covered"]
        .agg(["sum", "count"])
        .reset_index()
        .assign(rate=lambda x: x["sum"] / x["count"])
    )
    axes[1, 0].bar(
        coverage["n_stations"].astype(str),
        coverage["rate"],
        color="#377eb8",
        alpha=0.82,
    )
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("直径圆能覆盖定位区域的频率")
    axes[1, 0].set_xlabel("检测点数量 n")
    axes[1, 0].set_ylabel("覆盖频率")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    sc = axes[1, 1].scatter(
        df["D"],
        df["r_mec"],
        c=df["n_stations"],
        cmap="viridis",
        s=12,
        alpha=0.55,
        linewidth=0.0,
    )
    diag = np.linspace(0.0, df["D"].max() * 1.05, 2)
    axes[1, 1].plot(diag, 0.5 * diag, color="red", linestyle="--", label="r_MEC = D/2")
    axes[1, 1].plot(diag, diag / math.sqrt(3.0), color="black", linestyle=":", label="Jung 上界")
    axes[1, 1].set_title("r_MEC 与 D 的关系")
    axes[1, 1].set_xlabel("D / m")
    axes[1, 1].set_ylabel("r_MEC / m")
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[1, 1], label="检测点数量 n")

    fig.suptitle("第一问覆盖判定的蒙特卡洛鲁棒性检验", y=1.02)
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"fig_q1_robustness.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    true_source, stations, bearings = make_synthetic_case()
    stations = np.asarray(stations, dtype=float)
    polygon, bounded = intersect_bearing_wedges(stations, bearings)
    if len(polygon) < 3:
        raise RuntimeError("The synthetic case produced an empty feasible region.")

    result = analyze_region(polygon)
    result.bounded_by_angle_only = bounded

    pd.DataFrame(
        {
            "vertex_id": range(1, len(result.hull) + 1),
            "x": result.hull[:, 0],
            "y": result.hull[:, 1],
        }
    ).to_csv(ROOT / "q1_vertices.csv", index=False, encoding="utf-8-sig")

    p, q = result.diameter_pair
    summary = pd.DataFrame(
        [
            {
                "D": result.diameter,
                "D_over_2": 0.5 * result.diameter,
                "r_mec": result.mec.radius,
                "kappa": result.mec.radius / (0.5 * result.diameter),
                "area": polygon_area(result.hull),
                "covered": result.covered,
                "bounded_by_angle_only": result.bounded_by_angle_only,
                "p_x": p[0],
                "p_y": p[1],
                "q_x": q[0],
                "q_y": q[1],
                "mec_center_x": result.mec.center[0],
                "mec_center_y": result.mec.center[1],
            }
        ]
    )
    summary.to_csv(ROOT / "q1_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    draw_region_analysis(ax, stations, bearings, true_source, result)

    x_vals = np.concatenate([stations[:, 0], [true_source[0]], result.hull[:, 0]])
    y_vals = np.concatenate([stations[:, 1], [true_source[1]], result.hull[:, 1]])
    margin = 1.35 * max(np.ptp(x_vals), np.ptp(y_vals)) + 120.0
    xc = 0.5 * (np.min(x_vals) + np.max(x_vals))
    yc = 0.5 * (np.min(y_vals) + np.max(y_vals))
    half = max(0.5 * np.ptp(x_vals), 0.5 * np.ptp(y_vals), 180.0)
    ax.set_xlim(xc - half - 0.2 * margin, xc + half + 0.2 * margin)
    ax.set_ylim(yc - half - 0.2 * margin, yc + half + 0.2 * margin)

    ax.set_title(
        f"交会定位区域、集合直径与覆盖判定\n"
        f"D={result.diameter:.4f} m, D/2={0.5*result.diameter:.4f} m, "
        f"r_MEC={result.mec.radius:.4f} m, 覆盖={result.covered}",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"fig_q1_geometry.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    draw_criterion_cases()
    mc_df = run_monte_carlo()
    draw_robustness(mc_df)

    print("Question 1 summary:")
    print(summary.to_string(index=False))
    print("\nMonte Carlo summary:")
    print(
        mc_df.groupby("n_stations")
        .agg(
            mean_D=("D", "mean"),
            mean_r_mec=("r_mec", "mean"),
            mean_kappa=("kappa", "mean"),
            coverage_rate=("covered", "mean"),
            samples=("covered", "size"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
