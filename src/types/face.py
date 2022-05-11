from __future__ import annotations

from ctypes import Union
from typing import Optional

from pydantic import BaseModel
from src.types.bbox import BBox


class Face(BBox):
    def intersect(self, box: Union[BBox]) -> Optional[Face]:

        if isinstance(box, BBox):
            return super().intersect(box)
        # TODO: Make intersection with face

        return None
