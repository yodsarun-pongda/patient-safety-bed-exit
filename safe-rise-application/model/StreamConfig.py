from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamConfig:
    # How many frames per second we try to send to the browser
    target_fps: float = 1
    jpeg_quality: int = 80
    loop_file: bool = True
    isCameraStream: bool = False
    local_video_source: Optional[str] = None

    # Optional resize before running YOLO (helps performance)
    resize_width: Optional[int] = 640
    resize_height: Optional[int] = None

    # YOLO model and confidence threshold
    yolo_pose_model: str = "yolov8n-pose.pt"
    conf: float = 0.35

    # Run inference every N frames (1 = every frame)
    infer_every_n_frames: int = 1

    # Overlay options for visualization
    draw_skeleton: bool = True
    draw_bbox: bool = True
    draw_fps: bool = True
    draw_status_text: bool = True
