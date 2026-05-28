"""
spline_to_step.py
Reads a CSV of X,Y,Z spline points and sweeps a circular cross-section
along the path to produce a solid curved cylinder, exported as STEP.

Requires:  pip install cadquery
"""

import csv
from pathlib import Path

import cadquery as cq

# ── Parameters ────────────────────────────────────────────────────────────────
INPUT_CSV   = "barre GG.csv"   # CSV with x,y,z header columns
OUTPUT_STEP = "barre GG.step"  # output filename
RADIUS      = 5.0              # tube radius (same units as the CSV)
DOWNSAMPLE  = 4                # keep every Nth point to avoid an overly dense spline
                               # (set to 1 to use all points)
# ─────────────────────────────────────────────────────────────────────────────


def read_points(csv_path: str) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
    if not pts:
        raise ValueError(f"No points found in {csv_path!r}")
    return pts


def make_pipe(
    points: list[tuple[float, float, float]],
    radius: float,
) -> cq.Workplane:
    vectors = [cq.Vector(*p) for p in points]

    # Build a smooth B-spline wire through all sample points
    # (spline lives on Edge in CadQuery; wrap it in a Wire for sweep)
    spline_edge = cq.Edge.makeSpline(vectors, periodic=False)
    path = cq.Wire.assembleEdges([spline_edge])
    # Profile plane: centred at the spline start, normal along the first tangent
    start   = path.startPoint()
    tangent = path.tangentAt(0)          # cq.Vector, parameter 0 = start
    plane   = cq.Plane(origin=start, normal=tangent)

    # Sweep a circle along the path (Frenet frame keeps the profile perpendicular)
    return cq.Workplane(plane).circle(radius).sweep(path, isFrenet=True)


def spline_to_step(
    points: list[tuple[float, float, float]],
    output_path: str,
    radius: float = 5.0,
    downsample: int = 4,
) -> None:
    """Export a list of (x, y, z) points as a swept-tube STEP file.

    Parameters
    ----------
    points      : dense point list along the spline path
    output_path : destination .step file
    radius      : tube cross-section radius (same units as points)
    downsample  : keep every Nth point to avoid an overly dense spline
    """
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to build a pipe.")
    if downsample > 1:
        last = pts[-1]
        pts = pts[::downsample]
        if pts[-1] != last:
            pts.append(last)
    pipe = make_pipe(pts, radius)
    cq.exporters.export(pipe, str(Path(output_path)))


def main() -> None:
    pts = read_points(INPUT_CSV)
    spline_to_step(pts, OUTPUT_STEP, radius=RADIUS, downsample=DOWNSAMPLE)
    print(f"STEP exported → {Path(OUTPUT_STEP).resolve()}")


if __name__ == "__main__":
    main()
