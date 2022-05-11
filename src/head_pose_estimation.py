import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore
from src.core.setting import settings


class HeadEstimation:
    def __init__(self):

        self.ie = IECore()
        self.network = self.ie.read_network(
            model=settings.HEAD_MODEL_XML_PATH, weights=settings.HEAD_MODEL_BIN_PATH
        )
        self.executable_network = self.ie.load_network(
            self.network, device_name="CPU", num_requests=1
        )
        self.input_name = next(iter(self.network.input_info))

    def preprocess_image(self, image: np.array):
        return cv.resize(
            src=image, dsize=settings.HEAD_INPUT_SIZE, interpolation=cv.INTER_AREA
        )

    def process_face(self, face_image: np.array):
        input_image = self.preprocess_image(image=face_image)
        input_image = input_image.transpose(2, 0, 1)

        nn_outputs = self.executable_network.infer({self.input_name: input_image})

        pitch = nn_outputs["angle_p_fc"][0][0]
        roll = nn_outputs["angle_r_fc"][0][0]
        yaw = nn_outputs["angle_y_fc"][0][0]

        return pitch, roll, yaw
