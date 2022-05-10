import cv2 as cv
import numpy as np

from src.face_detector import FaceDetector
from src.face_reid import FaceRaid
from src.types.face import Face
from src.head_pose_estimation import HeadEstimation
from src.types.bbox import BBox


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


def main():
    vid = cv.VideoCapture(0)
    detector = FaceDetector()
    # reid = FaceRaid()
    head_estimator = HeadEstimation()

    while True:
        _, frame = vid.read()

        zone_face = create_image_with_face_zone(frame)
        faces: list[Face] = detector.process_frame(frame)
        bbox = get_ellipse_bbox(frame)

        zone_face = cv.rectangle(
            zone_face,
            pt1=(bbox.x_min, bbox.y_min),
            pt2=(bbox.x_max, bbox.y_max),
            color=(0, 0, 255),
            thickness=3,
        )

        for face in faces:
            # TODO: Add zone and intersection checking
            cv.rectangle(
                img=frame,
                pt1=face.get_min_point(),
                pt2=face.get_max_point(),
                color=(0, 0, 255),
            )

            face_image = frame[face.y_min : face.y_max, face.x_min : face.x_max]
            head_estimator.process_face(face_image=face_image)

        cv.imshow("WebCam", zone_face)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
