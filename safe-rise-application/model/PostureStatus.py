from dataclasses import dataclass, field
import time


@dataclass
class PostureStatus:
    message: str = "starting"
    posture: str = "unknown"
    person_count: int = 0
    alert_seq: int = 0
    alert_message: str = ""
    det_score: float = 0.0
    pose_score: float = 0.0
    updated_at: float = field(default_factory=time.time)
