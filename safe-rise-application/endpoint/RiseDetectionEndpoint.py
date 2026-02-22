import time
from fastapi import APIRouter

from service.PostureDetectionService import postureDetectionService
from model.StreamConfig import StreamConfig

router = APIRouter()

# Default config for pose streaming
cfg = StreamConfig(
    resize_width=640,
    yolo_pose_model="yolov8n-pose.pt",
    conf=0.35,
    infer_every_n_frames=1,
    target_fps=15,
    jpeg_quality=80,
)

# Change source to your RTSP URL if needed
# postureDetectionService = PostureDetectionService(source=0, config=cfg)

# Build response as video
@router.get("/video")
def video():
    return postureDetectionService.mjpeg_response()

# Build response status
@router.get("/status")
def status():
    if hasattr(postureDetectionService, "get_status") and callable(getattr(postureDetectionService, "get_status")):
        data = postureDetectionService.get_status() or {}
        # ensure updated_at exists
        data.setdefault("updated_at", int(time.time() * 1000))
        return data

    # Otherwise, build a payload from common attribute names.
    now_ms = int(time.time() * 1000)

    posture = getattr(postureDetectionService, "posture", None) or getattr(postureDetectionService, "last_posture", None)
    det_score = getattr(postureDetectionService, "det_score", None) or getattr(postureDetectionService, "last_det_score", None)
    pose_score = getattr(postureDetectionService, "pose_score", None) or getattr(postureDetectionService, "last_pose_score", None)
    person_count = getattr(postureDetectionService, "person_count", None) or getattr(postureDetectionService, "last_person_count", None)

    message = getattr(postureDetectionService, "message", None) or getattr(postureDetectionService, "last_message", None)

    updated_at = (
        getattr(postureDetectionService, "updated_at", None)
        or getattr(postureDetectionService, "last_updated_at", None)
        or now_ms
    )

    alert_message = getattr(postureDetectionService, "alert_message", None) or getattr(postureDetectionService, "last_alert_message", None)
    alert_seq = getattr(postureDetectionService, "alert_seq", None) or getattr(postureDetectionService, "last_alert_seq", None)

    payload = {
        "posture": posture if posture else "SIT",
        "det_score": det_score,
        "pose_score": pose_score,
        "person_count": person_count,
        "message": message,
        "updated_at": int(updated_at) if isinstance(updated_at, (int, float)) else now_ms,
        "alert_message": alert_message,
        "alert_seq": int(alert_seq) if isinstance(alert_seq, (int, float)) else alert_seq,
    }

    # Drop keys with None to keep the payload clean (matches optional fields)
    return {k: v for k, v in payload.items() if v is not None}