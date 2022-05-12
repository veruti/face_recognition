import math

import cv2 as cv
import numpy as np

from src.core.setting import settings
from src.db.faiss import IndexRepository
from src.face_detector import FaceDetector
from src.face_reid import FaceRaid
from src.head_pose_estimation import HeadEstimation
from src.types.bbox import BBox
from src.types.face import Face
from src.utils.draw import draw_bbox, draw_head_position_with_box


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
    index_repo = IndexRepository()
    detector = FaceDetector()
    reid = FaceRaid()
    head_estimator = HeadEstimation()

    vid = cv.VideoCapture(0)
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
            index_repo.search_vector(np.array([vector], dtype=np.float32))
            # TODO: print some results to image

        cv.imshow("WebCam", working_image)

        #  STOP CONDITIONS
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv.destroyAllWindows()
    index_repo.save_index()


if __name__ == "__main__":
    main()
