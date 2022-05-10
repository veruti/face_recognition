import cv2 as cv

from src.face_detector import FaceDetector
from src.face_reid import FaceRaid
from src.types.face import Face
from src.head_pose_estimation import HeadEstimation


def main():
    vid = cv.VideoCapture(0)
    detector = FaceDetector()
    reid = FaceRaid()
    head_estimator = HeadEstimation()

    while True:
        ret, frame = vid.read()
        faces: list[Face] = detector.process_frame(frame)

        for face in faces:
            cv.rectangle(
                img=frame,
                pt1=(face.x_min, face.y_min),
                pt2=(face.x_max, face.y_max),
                color=(0, 0, 255),
            )

            face_image = frame[face.y_min : face.y_max, face.x_min : face.x_max]
            # reid.process_face(face_image=face_image)
            head_estimator.process_face(face_image=face_image)

        cv.imshow("WebCam", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
