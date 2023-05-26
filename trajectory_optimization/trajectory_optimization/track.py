import logging
from typing import List

import ezdxf
import numpy as np
import shapely.geometry as geom

from trajectory_optimization import data_dir


class Track:
    def __init__(
        self,
        filepath=data_dir / "tracks/vivatech_2023.dxf",
        obstacles_filepath=data_dir / "tracks/vivatech_2023_obstacles.dxf",
    ):
        self._parse_track(filepath)

        if obstacles_filepath is not None:
            self._parse_obstacles(
                obstacles_filepath
            )  # coodinates system needs to line up w/ the track
        else:
            self.obstacles = []

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
        self.asphalt = geom.Polygon(shell=self.outer, holes=[self.inner])

    def _parse_obstacles(self, filepath):
        self.obstacles = self._parse_dxf(filepath)
        self.obstacles = [geom.Polygon(linear_ring) for linear_ring in self.obstacles]

    def is_inside(self, x, y):
        """given x, y coords, check if they are inside the track (between inner and outer)"""
        point = geom.Point(x, y)
        return self.asphalt.contains(point)  # and not self.outer.contains(point)

    def get_progress(self, x, y):
        """project x,y on centerline and get track progress from the start of the centerline."""
        point = geom.Point(x, y)
        return self.center.project(point)

    def _get_distances_to_objects(self, x, y, yaw, objects, angles):
        """given the car coordinates and orientation raymarch to objects (at different agnles) and
        return the distance kind of like lidar."""
        origin = geom.Point(x, y)
        lidar = {}
        for a in angles:
            # Calculate end point of the ray
            a = np.radians(a)
            theta = a + yaw
            end_x = x + 1000 * np.cos(theta)
            end_y = y + 1000 * np.sin(theta)
            line = geom.LineString([(x, y), (end_x, end_y)])

            # Calculate intersections with objects (track lines or obstacles)
            intersections = [line.intersection(o) for o in objects]

            # Get distances to the intersections and store the minimum distance
            distances = [
                origin.distance(intersection) if intersection else np.inf
                for intersection in intersections
            ]
            lidar[a] = min(distances)

        return lidar

    def get_distances_to_sides(self, x, y, yaw, angles=[-90, -45, 0, 45, 90]):
        objects = [self.inner, self.outer]

        return self._get_distances_to_objects(x, y, yaw, objects, angles)

    def get_distances_to_obstacles(self, x, y, yaw, angles=[-90, -45, 0, 45, 90]):
        return self._get_distances_to_objects(x, y, yaw, self.obstacles, angles)


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
