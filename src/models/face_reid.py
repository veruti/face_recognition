import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore

from src.core.setting import settings


class FaceRaid:
    def __init__(self):
        self.ie = IECore()
        self.network = self.ie.read_network(
            model=settings.REID_MODEL_XML_PATH, weights=settings.REID_MODEL_BIN_PATH
        )
        self.executable_network = self.ie.load_network(
            self.network, device_name="CPU", num_requests=1
        )
        self.input_name = next(iter(self.network.input_info))

    @staticmethod
    def preprocess_image(image: np.array):
        return cv.resize(
            src=image, dsize=settings.REID_INPUT_SIZE, interpolation=cv.INTER_AREA
        )

    def get_face_vector(self, face_image: np.array):
        preprocessed_image = self.preprocess_image(image=face_image)
        preprocessed_image = preprocessed_image.transpose(2, 0, 1)

        nn_outputs = self.executable_network.infer(
            {self.input_name: preprocessed_image}
        )
        return nn_outputs["658"].reshape((256,)).tolist()
