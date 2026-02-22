
from __future__ import annotations

import time
import threading
from dataclasses import asdict
from typing import Any, Dict, Generator, Optional, Union

import cv2
from model.PostureStatus import PostureStatus
from model.StreamConfig import StreamConfig

try:
    from starlette.responses import StreamingResponse
except Exception:  # pragma: no cover
    StreamingResponse = None  # type: ignore

# Key point
from model.PoseConfig import map_keypoints_xy_conf, classify_posture_coco17

# Import CV
import cv2

from service.PostureDetectionService import postureDetectionService

def show_img(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_image(imgPath: str, isShowImg: bool = False):
    img = cv2.imread(imgPath)

    # Show read image
    if img is not None and isShowImg:
        show_img(img)
    return img
    
def parse_to_yolo(img):
    annotated, person_count, posture, det_score, pose_score = postureDetectionService._acquire_pose(img)
    show_img(annotated)
    print("Classification by YOLO {}")

def classification_pose():
    print("Classification pose")

if __name__ == "__main__":
    # Read image
    img = read_image("test-img/IMG_2530_00000030.jpg", False)
    
    # Classification And Draw yolo
    parse_to_yolo(img)

    # draw_point()
    # classification_pose()
    print("Hello world")