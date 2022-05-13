from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BBox(BaseModel):
    x_min: int
    x_max: int

    y_max: int
    y_min: int

    @property
    def center_point(self) -> tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def min_point(self) -> tuple[int, int]:
        return (self.x_min, self.y_min)

    @property
    def max_point(self) -> tuple[int, int]:
        return (self.x_max, self.y_max)

    def intersect(self, box: BBox) -> Optional[BBox]:
        x_max = min(self.x_max, box.x_max)
        x_min = max(self.x_min, box.x_min)
        y_max = min(self.y_max, box.y_max)
        y_min = max(self.y_min, box.y_min)

        if (x_max <= x_min) or (y_max <= y_min):
            return None

        return BBox(x_min=x_min, x_max=x_max, y_max=y_max, y_min=y_min)

    def iou(self, box: BBox):
        x_max = min(self.x_max, box.x_max)
        x_min = max(self.x_min, box.x_min)
        y_max = min(self.y_max, box.y_max)
        y_min = max(self.y_min, box.y_min)

        if (x_max <= x_min) or (y_max <= y_min):
            return 0

        area_1 = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        area_2 = (box.x_max - box.x_min) * (box.y_max - box.y_min)
        intersection_area = (x_max - x_min) * (y_max - y_min)

        return intersection_area / (area_1 + area_2 - intersection_area)
