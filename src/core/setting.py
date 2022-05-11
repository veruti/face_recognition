from telnetlib import SE
from pydantic import BaseConfig


class Settings(BaseConfig):
    # DETECTION SETTINGS
    DETECTION_THRESHOLD = 0.9
    DETECTION_INPUT_SIZE = (448, 448)
    DETECTION_MODEL_XML_PATH = (
        "models/face-detector/face-detection-0204/face-detection-0204.xml"
    )
    DETECTION_MODEL_BIN_PATH = (
        "models/face-detector/face-detection-0204/face-detection-0204.bin"
    )

    # REID SETTINGS
    REID_MODEL_XML_PATH = "models/face-reidentification/face-reidentification-retail-0095/face-reidentification-retail-0095.xml"
    REID_MODEL_BIN_PATH = "models/face-reidentification/face-reidentification-retail-0095/face-reidentification-retail-0095.bin"

    REID_INPUT_SIZE = (128, 128)

    # HEAD POSE ESTIMATION
    HEAD_MODEL_XML_PATH = "models/head-pose-estimation/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml"
    HEAD_MODEL_BIN_PATH = "models/head-pose-estimation/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.bin"

    HEAD_INPUT_SIZE = (60, 60)

    # ADD FACE STAGE SETTINGS
    IOU_THRESHOLD = 0

    YAW_MIN_THRESHOLD = -10
    YAW_MAX_THRESHOLD = 10

    PITCH_MIN_THRESHOLD = -10
    PITCH_MAX_THRESHOLD = 10


settings = Settings()
