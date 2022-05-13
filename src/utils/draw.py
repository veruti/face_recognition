from typing import Union

import cv2 as cv
import numpy as np

from src.types.bbox import BBox
from src.types.face import Face


def draw_bbox(
    image: np.array, bbox: Union[BBox, Face], color: tuple[int, int, int], thickness=3
):
    return cv.rectangle(
        img=image,
        pt1=bbox.min_point,
        pt2=bbox.max_point,
        color=color,
        thickness=thickness,
    )


def draw_head_position_with_box(
    image: np.array, bbox: Union[BBox, Face], pitch: float, roll: float, yaw: float
) -> np.array:
    x_c, y_c = bbox.center_point

    d = 40
    x_roll, y_roll = int(x_c + np.sin(roll) * d), int(y_c - np.cos(roll) * d)
    x_yaw, y_yaw = int(x_c + d * yaw * 180 / 3.14 / 50), y_c
    x_pitch, y_pitch = x_c, int(y_c + d * pitch * 180 / 3.14 / 70)

    new_image = cv.line(
        image, pt1=(x_c, y_c), pt2=(x_roll, y_roll), color=(0, 255, 0), thickness=4
    )
    new_image = cv.line(
        new_image, pt1=(x_c, y_c), pt2=(x_yaw, y_yaw), color=(255, 0, 0), thickness=4
    )
    new_image = cv.line(
        new_image,
        pt1=(x_c, y_c),
        pt2=(x_pitch, y_pitch),
        color=(0, 0, 255),
        thickness=4,
    )
    return new_image
