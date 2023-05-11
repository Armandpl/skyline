import ezdxf
import numpy as np
import shapely.geometry as geom


class Track:
    def __init__(self, filepath="../data/tracks/vivatech_2023.dxf"):
        self._parse_track(filepath)

    def _parse_track(self, filepath):
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        lines = sort_shapes(list(msp))
        lines.sort(key=lambda x: x.length)

        # inner < center < outer
        self.inner, self.center, self.outer = lines
        self.asphalt = geom.Polygon(shell=self.outer, holes=[self.inner])
        self.center_polygon = geom.Polygon(
            shell=self.center
        )  # to know which side of the road we're on

        # TODO work out if the track is CCW or CW, reverse init state of the car based on that?
        # https://shapely.readthedocs.io/en/stable/reference/shapely.is_ccw.html
        # or maybe reverse the linear ring?

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


def discretize_arc(center, radius, start_angle, end_angle, num_segments=1000):
    # https://stackoverflow.com/questions/30762329/how-to-create-polygons-with-arcs-in-shapely-or-a-better-library
    centerx, centery = center

    # The coordinates of the arc
    theta = np.radians(np.linspace(start_angle, end_angle, num_segments))
    x = centerx + radius * np.cos(theta)
    y = centery + radius * np.sin(theta)

    return np.column_stack([x, y])


def dxf_arc_to_points(dxf_entity):
    d = dxf_entity.dxf
    x = d.center.x
    y = d.center.y
    return discretize_arc(
        center=(x, y),
        radius=d.radius,
        start_angle=d.start_angle,
        end_angle=d.end_angle,
    )


def dxf_line_to_points(dxf_entity):
    start = dxf_entity.dxf.start
    end = dxf_entity.dxf.end
    return np.array([[start.x, start.y], [end.x, end.y]])


def dxf_to_points(dxf_entity):
    if isinstance(dxf_entity, ezdxf.entities.Line):
        return dxf_line_to_points(dxf_entity)
    elif isinstance(dxf_entity, ezdxf.entities.Arc):
        return dxf_arc_to_points(dxf_entity)


def sort_shapes(elements):
    """takes in a list of ezdxf lines and arcs seperate out the inner, center, outer track lines
    return shapely linearings."""
    shapes = []
    current_shape = None

    def get_start(dxf_entity):
        if isinstance(dxf_entity, ezdxf.entities.Arc):
            return dxf_entity.start_point
        if isinstance(dxf_entity, ezdxf.entities.Line):
            return dxf_entity.dxf.start

    def get_end(dxf_entity):
        if isinstance(dxf_entity, ezdxf.entities.Arc):
            return dxf_entity.end_point
        if isinstance(dxf_entity, ezdxf.entities.Line):
            return dxf_entity.dxf.end

    def find_next(last_point):
        for i, element in enumerate(elements):
            # get both end of the line/arc
            start = get_start(element)
            end = get_end(element)
            start = np.array([start.x, start.y])
            end = np.array([end.x, end.y])

            if np.isclose(start, last_point, atol=1e-1).all():
                return (i, dxf_to_points(element))
            elif np.isclose(end, last_point, atol=1e-1).all():  # reversed
                points = dxf_to_points(element)
                return (i, np.flip(points, axis=0))

        print("can't find next, impossible bc closed shapes")
        raise RuntimeError

    while elements:  # while we havent processed all elements
        if current_shape is None:  # if we haven't started processing a shape
            # take a random element, convert to points, put it in the shape
            current_shape = dxf_to_points(elements.pop(0))
        else:  # if we started
            idx, next = find_next(current_shape[-1])
            current_shape = np.concatenate([current_shape, next], axis=0)
            del elements[idx]

            # check if the shape is closed
            if np.isclose(current_shape[0], current_shape[-1], atol=1e-1).all():
                shapes.append(geom.LinearRing(current_shape))
                current_shape = None

    return shapes
