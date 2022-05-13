import math

import cv2 as cv
import numpy as np

from src.core.setting import settings
from src.db.faiss import IndexRepository
from src.models.face_detector import FaceDetector
from src.models.face_reid import FaceRaid
from src.models.head_pose_estimation import HeadEstimation
from src.types.face import Face
from src.utils.draw import draw_bbox, draw_head_position_with_box


def main():
    index_repo = IndexRepository()

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

    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        working_image = frame.copy()
        faces: list[Face] = detector.get_faces(frame)

        for face in faces:
            working_image = draw_bbox(image=working_image, bbox=face, color=(255, 0, 0))
            face_image = frame[face.y_min : face.y_max, face.x_min : face.x_max]

            pitch, roll, yaw = head_estimator.get_face_rotations(face_image=face_image)

            working_image = draw_head_position_with_box(
                image=working_image,
                bbox=face,
                pitch=math.radians(pitch),
                roll=math.radians(roll),
                yaw=math.radians(yaw),
            )

            vector = reid.get_face_vector(face_image=face_image)
            if (
                (10 <= pitch)
                and (-10 < yaw < 10)
                and (len(face_container["up"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["up"].append(vector)
            elif (
                (10 <= pitch)
                and (yaw <= -10)
                and (len(face_container["up_left"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["up_left"].append(vector)
            elif (
                (10 <= pitch)
                and (10 <= yaw)
                and (len(face_container["up_right"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["up_right"].append(vector)

            elif (
                (-10 < pitch < 10)
                and (-10 < yaw < 10)
                and (len(face_container["middle"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["middle"].append(vector)

            elif (
                (-10 < pitch < 10)
                and (yaw <= -10)
                and (len(face_container["middle_left"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["middle_left"].append(vector)

            elif (
                (-10 < pitch < 10)
                and (10 <= yaw)
                and (
                    len(face_container["middle_right"]) < settings.MAX_VECTORS_PER_SIDE
                )
            ):
                face_container["middle_right"].append(vector)

            elif (
                (pitch < -10)
                and (-10 <= yaw <= 10)
                and (len(face_container["down"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["down"].append(vector)

            elif (
                (pitch < -10)
                and (yaw <= -10)
                and (len(face_container["down_left"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["down_left"].append(vector)

            elif (
                (10 <= pitch)
                and (10 <= yaw)
                and (len(face_container["down_right"]) < settings.MAX_VECTORS_PER_SIDE)
            ):
                face_container["down_right"].append(vector)

            print(sum([len(face_container[key]) for key in face_container]))

        cv.imshow("WebCam", working_image)

        #  STOP CONDITIONS
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        if (
            sum([len(face_container[key]) for key in face_container])
            == 9 * settings.MAX_VECTORS_PER_SIDE
        ):
            for key in face_container:
                vectors = np.vstack(face_container[key]).astype(np.float32)
                index_repo.add_vectors(vectors=vectors)
            break

    vid.release()
    cv.destroyAllWindows()
    index_repo.save_index()


if __name__ == "__main__":
    main()
