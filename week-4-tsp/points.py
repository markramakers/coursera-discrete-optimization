import math
from collections import namedtuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    index: int = 0

    def to_list(self):
        return [self.x, self.y]


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
