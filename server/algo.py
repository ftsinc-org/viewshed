from __future__ import division

import math
import itertools
import heapq
from enum import Enum
import pdb
import shapely.geometry
import shapely.ops
from functools import total_ordering

SPACING = 3


def pixel_angle(pixel, center_pixel):
    x, y = pixel
    center_x, center_y = center_pixel
    angle = math.atan2(x - center_x, y - center_y)
    if angle < 0:
        return angle + 2 * math.pi
    return angle


# We define the center of the pixel to be its coordinates.  Thus pixel (x, y)
# is a square with bottom-left corner (x - .5, y - .5) and top-right corner
# (x + .5, y + .5).
def pixel_angles(pixel, center_pixel):
    """Generate the angles for a given pixel.

    Args:
      pixel: The pixel to generate the angles for as a tuple.
      center_pixel: The center pixel as a tuple.
    Returns:
      A dict containing 'center', 'min', and 'max'.  Each with an angle in radians.
    """
    x, y = pixel
    corners = [(x + i * 0.5, y + j * 0.5) for i in [1, -1] for j in [1, -1]]
    min_angle = min([(pixel_angle(corner, center_pixel), corner) for corner in corners])
    max_angle = max([(pixel_angle(corner, center_pixel), corner) for corner in corners])
    return {
        "center": (pixel_angle(pixel, center_pixel), pixel),
        "min": (min_angle[0], min_angle[1]),
        "max": (max_angle[0], max_angle[1]),
    }


class EventState(Enum):
    START = 1
    CENTER = 2
    END = 3


@total_ordering
class PixelEvent(object):
    def __init__(self, coordinate, pixel, event_state, center_pixel):
        self.coordinate = coordinate
        self.pixel = pixel
        self.event_state = event_state
        self.center_pixel = center_pixel

    def __lt__(self, other):
        return (
            pixel_angle(self.coordinate, self.center_pixel),
            pixel_distance(self.pixel, self.center_pixel),
            self.event_state.value,
        ) < (
            pixel_angle(other.coordinate, other.center_pixel),
            pixel_distance(other.pixel, other.center_pixel),
            other.event_state.value,
        )

    def __hash__(self):
        return hash((self.coordinate, self.pixel, self.event_state, self.center_pixel))

    def __eq__(self, other):
        return self.pixel == other.pixel and self.event_state == other.event_state

    def __str__(self):
        return (
            "PixelEvent{pixel="
            + str(self.pixel)
            + ", event_state="
            + self.event_state.name
            + "}"
        )


class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.elements = set()

    def push(self, element):
        if not element in self.elements:
            heapq.heappush(self.heap, element)
            self.elements.add(element)

    def pop(self):
        element = heapq.heappop(self.heap)
        # Don't remove the element from the elements set to avoid duplicates
        # self.elements.remove(element)
        return element

    def is_empty(self):
        return len(self.heap) == 0


def pixel_distance(a, b):
    """Returns the distance in pixels between a and b"""
    x, y = b[0] - a[0], b[1] - a[1]
    return math.sqrt(x * x + y * y)


def add_surrounding_pixels(pixel, center_pixel, radius, priority_set):
    x, y = pixel
    pixels = [
        (x + i, y + j)
        for i in [1, 0, -1]
        for j in [1, 0, -1]
        if not (i == 0 and j == 0)
    ]
    current_angle = pixel_angle(pixel, center_pixel)
    for new_pixel in pixels:
        angle = pixel_angle(new_pixel, center_pixel)
        # Angles should be in the range [0, 2 * pi], which will cause
        # the queue to eventually be empty.
        if angle <= current_angle or pixel_distance(new_pixel, center_pixel) > radius:
            continue
        angles = pixel_angles(new_pixel, center_pixel)
        start_event = PixelEvent(
            angles["min"][1], new_pixel, EventState.START, center_pixel
        )
        center_event = PixelEvent(
            angles["center"][1], new_pixel, EventState.CENTER, center_pixel
        )
        end_event = PixelEvent(
            angles["max"][1], new_pixel, EventState.END, center_pixel
        )
        priority_set.push(start_event)
        priority_set.push(center_event)
        priority_set.push(end_event)


def pixel_gradient(height, tower_height, distance):
    return math.atan2(distance, tower_height - height)


class GradientTree(object):
    def create_empty_tree(self, height):
        if height == 0:
            return [-math.inf]
        return [
            -math.inf,
            self.create_empty_tree(height - 1),
            self.create_empty_tree(height - 1),
        ]

    def add_gradient(self, gradient, distance, tree=None, radius=None, height=None):
        if tree == None:
            self.tree = self.add_gradient(
                gradient, distance, self.tree, self.radius, self.tree_height
            )
            return self.tree
        if height == 0:
            tree.append(gradient)
            return tree
        left_tree = tree[1]
        right_tree = tree[2]
        if distance < radius / 2:
            new_val = tree[0]
            if gradient > tree[0]:
                new_val = gradient
            return [
                new_val,
                self.add_gradient(
                    gradient, distance, left_tree, radius / 2, height - 1
                ),
                right_tree,
            ]
        else:
            new_val = tree[0]
            if gradient > tree[0]:
                new_val = gradient
            return [
                new_val,
                left_tree,
                self.add_gradient(
                    gradient, distance - radius / 2, right_tree, radius / 2, height - 1
                ),
            ]

    def search_for_max_gradient(self, distance, tree=None, radius=None, height=None):
        if tree == None:
            tree = self.tree
        if radius == None:
            radius = self.radius
        if height == None:
            height = self.tree_height
        if height == 0:
            return -math.inf
        if distance < radius / 2:
            return self.search_for_max_gradient(
                distance, tree[1], radius / 2, height - 1
            )
        return max(
            tree[1][0],
            self.search_for_max_gradient(
                distance - radius / 2, tree[2], radius / 2, height - 1
            ),
        )

    def remove_gradient(self, gradient, distance, tree=None, radius=None, height=None):
        if tree == None:
            self.tree = self.remove_gradient(
                gradient, distance, self.tree, self.radius, self.tree_height
            )
            return self.tree
        if height == 0:
            tree.remove(gradient)
            return tree
        left_tree = tree[1]
        right_tree = tree[2]
        if distance < radius / 2:
            left_tree = self.remove_gradient(
                gradient, distance, left_tree, radius / 2, height - 1
            )
        else:
            right_tree = self.remove_gradient(
                gradient, distance - radius / 2, right_tree, radius / 2, height - 1
            )
        if height == 1:
            return [max(max(left_tree), max(right_tree)), left_tree, right_tree]
        return [max(left_tree[0], right_tree[0]), left_tree, right_tree]

    def add_pixel(self, pixel, height):
        distance = pixel_distance(pixel, self.center_pixel)
        return self.add_gradient(
            pixel_gradient(height, self.tower_height, distance), distance
        )

    def remove_pixel(self, pixel, height):
        distance = pixel_distance(pixel, self.center_pixel)
        return self.remove_gradient(
            pixel_gradient(height, self.tower_height, distance), distance
        )

    def is_visible(self, pixel, height):
        distance = pixel_distance(pixel, self.center_pixel)
        gradient = pixel_gradient(height, self.tower_height, distance)
        if gradient > self.search_for_max_gradient(distance):
            return True

    def __init__(self, center_pixel, radius, tower_height):
        self.center_pixel = center_pixel
        self.tower_height = tower_height
        self.radius = radius
        self.tree_height = int(math.log(radius, 2)) + 1
        self.tree = self.create_empty_tree(self.tree_height)


async def find_visible_pixels(center_pixel, radius, tower_height, pixel_to_height_fn):
    priority_set = PrioritySet()
    tree = GradientTree(center_pixel, radius, tower_height)
    # Initialize the priority set.
    for i in range(round(radius)):
        coordinate = (center_pixel[0], center_pixel[1] + i + 1)
        pixel_event = PixelEvent(coordinate, coordinate, EventState.START, center_pixel)
        priority_set.push(pixel_event)
        pixel_event = PixelEvent(
            coordinate, coordinate, EventState.CENTER, center_pixel
        )
        priority_set.push(pixel_event)
        end_coordinate = (center_pixel[0] + 0.5, center_pixel[1] + i + 1)
        pixel_event = PixelEvent(
            end_coordinate,
            coordinate,
            EventState.END,
            center_pixel,
        )
        priority_set.push(pixel_event)

    while not priority_set.is_empty():
        event = priority_set.pop()
        pixel = event.pixel
        height = await pixel_to_height_fn(pixel)
        if event.event_state == EventState.START:
            tree.add_pixel(pixel, height)
        elif event.event_state == EventState.END:
            tree.remove_pixel(pixel, height)
            add_surrounding_pixels(pixel, center_pixel, radius, priority_set)
        elif event.event_state == EventState.CENTER:
            if tree.is_visible(pixel, height):
                yield pixel


async def visible_pixels_to_polygons(pixel_generator):
    polygons = [
        shapely.geometry.Polygon(
            [
                shapely.geometry.Point(x + 0.5 * i, y + 0.5 * j)
                for (i, j) in [(1, 1), (1, -1), (-1, -1), (-1, 1)]
            ]
        )
        async for (x, y) in pixel_generator
    ]
    return shapely.ops.unary_union(polygons)


async def find_visible_polygons(center_pixel, radius, tower_height, pixel_to_height_fn):
    return await visible_pixels_to_polygons(
        find_visible_pixels(center_pixel, radius, tower_height, pixel_to_height_fn)
    )


if __name__ == "__main__":
    assert iter_to_runs([False, False, True, True, False, True, False, True, True]) == [
        (2, 3),
        (5, 5),
        (7, 8),
    ]
    assert iter_to_runs([True]) == [(0, 0)]
    assert iter_to_runs([True, True, True, True, False, True, True]) == [(0, 3), (5, 6)]

    import matplotlib.pyplot as plt

    heightmap = [math.sin(x / 15.0) * x for x in xrange(360)]
    tower_height = 100.0  # foots above MSL

    filt = ray(tower_height, heightmap)

    fhm = [h if fl else 0 for (h, fl) in zip(heightmap, filt)]

    plt.scatter(range(len(heightmap)), fhm)
    plt.scatter([0], [tower_height], color="red")
    plt.plot(heightmap)
    plt.show()
