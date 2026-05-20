import logging
from typing import List

import ezdxf
import numpy as np
import shapely.geometry as geom
from shapely.ops import unary_union
from shapely.prepared import prep

from trajectory_optimization import data_dir

# dxf tracks are sampled at 0.5 deg/segment (~2700 pts/ring), which makes the
# lidar ray intersections needlessly slow. Simplifying to 1cm keeps the geometry
# within real-lidar noise (mean error <0.5cm) while cutting ray cost ~3x.
SIMPLIFY_TOLERANCE = 0.01


class Track:
    def __init__(
        self,
        filepath,
        obstacles_filepath=None,
    ):
        self._parse_track(filepath)

        self.obstacles = []
        self.obstacles_union = None
        self._obstacle_segments = _to_segments([])
        if obstacles_filepath is not None:
            self._parse_obstacles(
                obstacles_filepath
            )  # coodinates system needs to line up w/ the track

    def _parse_dxf(self, filepath):
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()

        linear_rings = []
        # each track line is a lwpolyline (https://ezdxf.readthedocs.io/en/stable/tutorials/lwpolyline.html)
        # or a circle, and we convert both to shapely linear rings
        for entity in list(msp):
            if isinstance(entity, ezdxf.entities.LWPolyline):
                linear_rings.append(convert_closed_lwpolyline(entity))
            elif isinstance(entity, ezdxf.entities.Circle):
                linear_rings.append(convert_circle(entity))
            else:
                logging.warning(f"Unexpected ezdxf entity: {type(entity)}, skipping")
            # TODO dxf is a bit finicky, maybe save tracks as np.arrays, might be more timeproof

        return linear_rings

    def _parse_track(self, filepath):
        lines = self._parse_dxf(filepath)
        lines.sort(key=lambda x: x.length)

        # inner < center < outer
        self.inner, self.center, self.outer = lines
        # simplify the boundaries used for lidar/inside checks (not the centerline,
        # which drives the progress reward and is left untouched)
        self.inner = self.inner.simplify(SIMPLIFY_TOLERANCE)
        self.outer = self.outer.simplify(SIMPLIFY_TOLERANCE)
        self.asphalt = geom.Polygon(shell=self.outer, holes=[self.inner])
        # precompute the boundary segments once for the vectorized lidar raycast
        self._side_segments = _to_segments([self.inner, self.outer])

    def _parse_obstacles(self, filepath):
        self.obstacles = [geom.Polygon(ring) for ring in self._parse_dxf(filepath)]
        # prepared union for fast "does the car overlap any obstacle" queries
        self.obstacles_union = prep(unary_union(self.obstacles))
        self._obstacle_segments = _to_segments([o.exterior for o in self.obstacles])

    def is_inside(self, x, y):
        """given x, y coords, check if they are inside the track (between inner and outer)"""
        point = geom.Point(x, y)
        return self.asphalt.contains(point)  # and not self.outer.contains(point)

    def get_progress(self, x, y):
        """project x,y on centerline and get track progress from the start of the centerline."""
        point = geom.Point(x, y)
        return self.center.project(point)

    def _raycast(self, x, y, yaw, angles, segments):
        """raymarch from the car (at each ray angle) to a set of segments and return the distance
        to the nearest forward hit, kind of like a lidar.

        Vectorized over (rays x segments): for each ray o + t*d and segment p1 + u*e we solve for
        (t, u) in closed form, keep hits with t >= 0 and u in [0, 1], and take the min t per ray.
        Returns a dict keyed by ray angle in radians (relative to the car), matching the renderer.
        """
        seg_start, edge = segments
        angles_rad = np.radians(np.asarray(angles, dtype=float))

        if seg_start.shape[0] == 0:
            return {a: np.inf for a in angles_rad}

        theta = angles_rad + yaw
        dx, dy = np.cos(theta), np.sin(theta)  # ray directions (unit), shape (R,)
        bx = seg_start[:, 0] - x  # (p1 - origin), shape (S,)
        by = seg_start[:, 1] - y
        ex, ey = edge[:, 0], edge[:, 1]  # segment vectors, shape (S,)

        with np.errstate(divide="ignore", invalid="ignore"):
            denom = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]  # d x e, (R, S)
            t = (bx * ey - by * ex)[None, :] / denom  # (b x e) / denom -> distance along ray
            u = (bx[None, :] * dy[:, None] - by[None, :] * dx[:, None]) / denom  # (b x d) / denom

        hit = (denom != 0) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        distances = t.min(axis=1)  # (R,)
        return {a: d for a, d in zip(angles_rad, distances)}

    def get_distances_to_sides(self, x, y, yaw, angles=[-90, -45, 0, 45, 90]):
        return self._raycast(x, y, yaw, angles, self._side_segments)

    def get_distances_to_obstacles(self, x, y, yaw, angles=[-90, -45, 0, 45, 90]):
        return self._raycast(x, y, yaw, angles, self._obstacle_segments)


def _to_segments(geoms):
    """flatten shapely line geometries into (start_points, edge_vectors) arrays of shape (S, 2),
    used by the vectorized lidar raycast."""
    starts, edges = [], []
    for g in geoms:
        coords = np.asarray(g.coords)
        starts.append(coords[:-1])
        edges.append(coords[1:] - coords[:-1])
    if not starts:
        empty = np.empty((0, 2))
        return empty, empty
    return np.concatenate(starts), np.concatenate(edges)


def convert_circle(
    circle: ezdxf.entities.Circle, degrees_per_segment: float = 0.5
) -> geom.LinearRing:

    circle = circle.dxf
    xy = arc_points(
        start_angle=0,
        end_angle=2 * np.pi,
        radius=circle.radius,
        center=circle.center,
        degrees_per_segment=degrees_per_segment,
    )

    return geom.LinearRing(xy)


def convert_closed_lwpolyline(
    polyline: ezdxf.entities.LWPolyline, degrees_per_segment: float = 0.5
) -> geom.LinearRing:
    """lwpolyline is a lightweight polyline (cf POLYLINE) only accept closed polylines, we only
    deal with closed racetracks modified from: https://github.com/aegis1980/cad-to-
    shapely/blob/master/cad_to_shapely/dxf.py."""
    assert polyline.closed, "polyline is not closed"

    xy = []

    points = polyline.get_points()

    for i, point in enumerate(points):
        x, y, _, _, b = point
        xy.append([x, y])

        if b != 0:  # if bulge
            # if next point is the end, next point is the start bc closed
            if i + 1 == len(points):
                next_point = points[0]
            else:
                next_point = points[i + 1]

            p1 = [x, y]
            p2 = [next_point[0], next_point[1]]

            pts = arc_points_from_bulge(p1, p2, b, degrees_per_segment)

            # exclude start and end points
            # start point was already added above
            # last point is next point, will be added when dealing with the next point
            pts = pts[1:-1]

            xy.extend(pts)

    return geom.LinearRing(xy)


def arc_points(
    start_angle: float,
    end_angle: float,
    radius: float,
    center: List[float],
    degrees_per_segment: float,
) -> list:
    """Coordinates of an arcs (for approximation as a polyline)

    Args:
        start_angle (float): arc start point relative to centre, in radians
        end_angle (float): arc end point relative to centre, in radians
        radius (float): [description]
        center (List[float]): arc centre as [x,y]
        degrees_per_segment (float): [description]

    Returns:
        list: 2D list of points as [x,y]

    from https://github.com/aegis1980/cad-to-shapely/blob/master/cad_to_shapely/utils.py
    """

    n = abs(int((end_angle - start_angle) / np.radians(degrees_per_segment)))  # number of segments
    theta = np.linspace(start_angle, end_angle, n)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return np.column_stack([x, y])


def distance(p1: List[float], p2: List[float]) -> float:
    """from https://github.com/aegis1980/cad-to-shapely/blob/master/cad_to_shapely/utils.py."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def arc_points_from_bulge(p1: List[float], p2: List[float], b: float, degrees_per_segment: float):
    """http://darrenirvine.blogspot.com/2015/08/polylines-radius-bulge-turnaround.html.

    Args:
        p1 (List[float]): [description]
        p2 (List[float]): [description]
        b (float): bulge of the arc
        degrees_per_segment (float): [description]

    Returns:
        [type]: point on arc

    from: https://github.com/aegis1980/cad-to-shapely/blob/master/cad_to_shapely/utils.py
    """

    theta = 4 * np.arctan(b)
    u = distance(p1, p2)

    r = u * ((b**2) + 1) / (4 * b)

    try:
        a = np.sqrt(r**2 - (u * u / 4))
    except ValueError:
        a = 0

    dx = (p2[0] - p1[0]) / u
    dy = (p2[1] - p1[1]) / u

    A = np.array(p1)
    B = np.array(p2)
    # normal direction
    N = np.array([dy, -dx])

    # if bulge is negative arc is clockwise
    # otherwise counter-clockwise
    s = b / abs(b)  # sigma = signum(b)

    # centre, as a np.array 2d point

    if abs(theta) <= np.pi:
        C = ((A + B) / 2) - s * a * N
    else:
        C = ((A + B) / 2) + s * a * N

    start_angle = np.arctan2(p1[1] - C[1], p1[0] - C[0])
    if b < 0:
        start_angle += np.pi

    end_angle = start_angle + theta

    return arc_points(start_angle, end_angle, r, C, degrees_per_segment)
