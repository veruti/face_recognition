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

        angle_p_fc = nn_outputs["angle_p_fc"]
        angle_r_fc = nn_outputs["angle_r_fc"]
        angle_y_fc = nn_outputs["angle_y_fc"]

        if 10 < abs(angle_r_fc):
            print("Take your head right")
        # print(f"p: {angle_p_fc}. r: {angle_r_fc}. y: {angle_y_fc}.")
