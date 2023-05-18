from typing import List

import ezdxf
import numpy as np
import shapely.geometry as geom


class Track:
    def __init__(self, filepath="../data/tracks/vivatech_2023.dxf"):
        self._parse_track(filepath)

    def _parse_track(self, filepath):
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()

        lines = []
        # each track line is a lwpolyline (https://ezdxf.readthedocs.io/en/stable/tutorials/lwpolyline.html)
        for lwpolyline in list(msp):
            # convert from exdxf entity to shapely linear ring
            lines.append(convert_closed_lwpolyline(lwpolyline))
            # TODO dxf is a bit finicky, maybe save tracks as np.arrays, might be more robust in time

        lines.sort(key=lambda x: x.length)

        # inner < center < outer
        self.inner, self.center, self.outer = lines
        self.asphalt = geom.Polygon(shell=self.outer, holes=[self.inner])
        self.center_polygon = geom.Polygon(
            shell=self.center
        )  # to know which side of the road we're on, # TODO delete now that we don't need pid anymore?

    def is_inside(self, x, y):
        """given x, y coords, check if they are inside the track (between inner and outer)"""
        point = geom.Point(x, y)
        return self.asphalt.contains(point)  # and not self.outer.contains(point)

    def get_progress(self, x, y):
        """project x,y on centerline and get track progress from the start of the centerline."""
        point = geom.Point(x, y)
        return self.center.project(point)

    def get_distance_to_side(self, x, y, angle, angles=[-90, -45, 0, 45, 90]):
        """given the car coordinates and orientation raymarch to the sides of the track (at
        different agnles) and return the distance kind of like lidar."""
        origin = geom.Point(x, y)
        distances = {}
        for a in angles:
            # Calculate end point of the ray
            a = np.radians(a)
            theta = a + angle
            end_x = x + 1000 * np.cos(theta)
            end_y = y + 1000 * np.sin(theta)
            line = geom.LineString([(x, y), (end_x, end_y)])

            # Calculate intersections with inner and outer lines
            inner_intersection = line.intersection(self.inner)
            outer_intersection = line.intersection(self.outer)

            # Get distances to the intersections and store the minimum distance
            inner_distance = origin.distance(inner_intersection) if inner_intersection else np.inf
            outer_distance = origin.distance(outer_intersection) if outer_intersection else np.inf
            distances[a] = min(inner_distance, outer_distance)

        return distances


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
