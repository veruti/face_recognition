import math
from pprint import pprint

import cv2 as cv
import numpy as np

from src.core.setting import settings
from src.face_detector import FaceDetector
from src.face_reid import FaceRaid
from src.head_pose_estimation import HeadEstimation
from src.types.bbox import BBox
from src.types.face import Face
from src.utils.draw import draw_bbox, draw_head_position_with_box
import json


def create_image_with_face_zone(image: np.array) -> np.array:
    face_zone = np.full(shape=image.shape, fill_value=127, dtype=np.uint8)

    face_zone = cv.ellipse(
        img=face_zone,
        center=(image.shape[1] // 2, image.shape[0] // 2),
        axes=(120, 150),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(255, 255, 255),
        thickness=-1,
    )

    return (image / 255) * (face_zone / 255)


def get_ellipse_bbox(image: np.array) -> BBox:
    y_c, x_c = image.shape[0] // 2, image.shape[1] // 2
    w_2, h_2 = 120, 150
    return BBox(x_min=x_c - w_2, y_min=y_c - h_2, x_max=x_c + w_2, y_max=y_c + h_2)


# TODO: Refactor Code and add vectors storage
def main():
    vid = cv.VideoCapture(0)
    detector = FaceDetector()
    reid = FaceRaid()
    head_estimator = HeadEstimation()

    face_container = {
        "up": [],
        "up_right": [],
        "up_left": [],
        "middle": [],
        "middle_right": [],
        "middle_left": [],
        "down": [],
        "down_left": [],
        "down_right": [],
    }

    max_vectors = 10

    while True:
        _, frame = vid.read()
        working_image = frame.copy()
        faces: list[Face] = detector.process_frame(frame)

        for face in faces:
            zone_face = draw_bbox(image=working_image, bbox=face, color=(255, 0, 0))
            face_image = frame[face.y_min : face.y_max, face.x_min : face.x_max]

            pitch, roll, yaw = head_estimator.process_face(face_image=face_image)

            zone_face = draw_head_position_with_box(
                image=zone_face,
                bbox=face,
                pitch=math.radians(pitch),
                roll=math.radians(roll),
                yaw=math.radians(yaw),
            )

            vector = reid.process_face(face_image=face_image)
            if (
                (10 <= pitch)
                and (-10 < yaw < 10)
                and (len(face_container["up"]) < max_vectors)
            ):
                face_container["up"].append(vector)
            elif (
                (10 <= pitch)
                and (yaw <= -10)
                and (len(face_container["up_left"]) < max_vectors)
            ):
                face_container["up_left"].append(vector)
            elif (
                (10 <= pitch)
                and (10 <= yaw)
                and (len(face_container["up_right"]) < max_vectors)
            ):
                face_container["up_right"].append(vector)

            elif (
                (-10 < pitch < 10)
                and (-10 < yaw < 10)
                and (len(face_container["middle"]) < max_vectors)
            ):
                face_container["middle"].append(vector)

            elif (
                (-10 < pitch < 10)
                and (yaw <= -10)
                and (len(face_container["middle_left"]) < max_vectors)
            ):
                face_container["middle_left"].append(vector)

            elif (
                (-10 < pitch < 10)
                and (10 <= yaw)
                and (len(face_container["middle_right"]) < max_vectors)
            ):
                face_container["middle_right"].append(vector)

            elif (
                (pitch < -10)
                and (-10 <= yaw <= 10)
                and (len(face_container["down"]) < max_vectors)
            ):
                face_container["down"].append(vector)

            elif (
                (pitch < -10)
                and (yaw <= -10)
                and (len(face_container["down_left"]) < max_vectors)
            ):
                face_container["down_left"].append(vector)

            elif (
                (10 <= pitch)
                and (10 <= yaw)
                and (len(face_container["down_right"]) < max_vectors)
            ):
                face_container["down_right"].append(vector)

            print(sum([len(face_container[key]) for key in face_container]))

        cv.imshow("WebCam", working_image)

        #  STOP CONDITIONS
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        if sum([len(face_container[key]) for key in face_container]) == 9 * max_vectors:
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
