import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore

from src.core.setting import settings
from src.types.face import Face


class FaceDetector:
    def __init__(self):
        self.ie = IECore()
        self.network = self.ie.read_network(
            model=settings.DETECTION_MODEL_XML_PATH,
            weights=settings.DETECTION_MODEL_BIN_PATH,
        )
        self.executable_network = self.ie.load_network(
            self.network, device_name="CPU", num_requests=1
        )

        self.input_name = next(iter(self.network.input_info))

    def get_faces(self, frame: np.array) -> list[Face]:
        frame_height, frame_width = frame.shape[:2]

        preprocessed = self.preprocess_image(image=frame)
        nn_outputs = self.executable_network.infer({self.input_name: preprocessed})
        faces_raw = nn_outputs["detection_out"][0][0]

        faces = []
        for (_, _, conf, x_min, y_min, x_max, y_max) in faces_raw:
            if settings.DETECTION_THRESHOLD <= conf:
                face = Face(
                    x_min=abs(int(x_min * frame_width)),
                    x_max=abs(int(x_max * frame_width)),
                    y_min=abs(int(y_min * frame_height)),
                    y_max=abs(int(y_max * frame_height)),
                )

                faces.append(face)

        return faces

    @staticmethod
    def preprocess_image(image: np.array):
        return cv.resize(
            src=image, dsize=settings.DETECTION_INPUT_SIZE, interpolation=cv.INTER_AREA
        ).transpose(2, 0, 1)
